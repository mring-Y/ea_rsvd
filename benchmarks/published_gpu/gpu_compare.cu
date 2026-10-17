// Complete sparse GPU SVD comparison. PRIMME is used through its public API;
// the solver algorithm is not reimplemented here. See THIRD_PARTY.md.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <curand.h>
#include <primme.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#define CUDA(x) do { auto e=(x); if(e!=cudaSuccess) throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(e)); } while(0)
#define BLAS(x) do { auto e=(x); if(e!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(#x)+": cuBLAS "+std::to_string(e)); } while(0)
#define SPARSE(x) do { auto e=(x); if(e!=CUSPARSE_STATUS_SUCCESS) throw std::runtime_error(std::string(#x)+": cuSPARSE "+std::to_string(e)); } while(0)
#define SOLVER(x) do { auto e=(x); if(e!=CUSOLVER_STATUS_SUCCESS) throw std::runtime_error(std::string(#x)+": cuSOLVER "+std::to_string(e)); } while(0)
#define RNG(x) do { auto e=(x); if(e!=CURAND_STATUS_SUCCESS) throw std::runtime_error(std::string(#x)+": cuRAND "+std::to_string(e)); } while(0)

using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }

template<class T> struct Device {
  T* ptr=nullptr;
  Device()=default;
  explicit Device(size_t count) { if(count) CUDA(cudaMalloc(reinterpret_cast<void**>(&ptr),count*sizeof(T))); }
  ~Device() { if(ptr) cudaFree(ptr); }
  Device(const Device&)=delete;
  Device& operator=(const Device&)=delete;
  Device(Device&& other) noexcept:ptr(other.ptr) { other.ptr=nullptr; }
  Device& operator=(Device&& other) noexcept { if(ptr) cudaFree(ptr); ptr=other.ptr; other.ptr=nullptr; return *this; }
};

template<class T> constexpr cudaDataType scalar_type() { return std::is_same<T,float>::value?CUDA_R_32F:CUDA_R_64F; }

struct Input {
  int m,n,nz;
  std::vector<int> offsets,columns;
  std::vector<double> values;
};

template<class T> void read_array(std::ifstream& f,T* p,size_t count) {
  f.read(reinterpret_cast<char*>(p),static_cast<std::streamsize>(count*sizeof(T)));
  if(!f) throw std::runtime_error("Truncated canonical CSR input");
}

Input load_input(const std::string& path) {
  std::ifstream f(path,std::ios::binary);
  if(!f) throw std::runtime_error("Cannot open CSR input");
  char magic[8]; uint64_t shape[3]; read_array(f,magic,8); read_array(f,shape,3);
  if(std::memcmp(magic,"PGCSR001",8) || !shape[0] || !shape[1] || !shape[2] ||
     *std::max_element(shape,shape+3)>=static_cast<uint64_t>(INT32_MAX))
    throw std::runtime_error("Invalid 32-bit canonical CSR header");
  Input a; a.m=static_cast<int>(shape[0]); a.n=static_cast<int>(shape[1]); a.nz=static_cast<int>(shape[2]);
  a.offsets.resize(a.m+1); a.columns.resize(a.nz); a.values.resize(a.nz);
  read_array(f,a.offsets.data(),a.offsets.size()); read_array(f,a.columns.data(),a.columns.size());
  read_array(f,a.values.data(),a.values.size());
  if(a.offsets.front()!=0 || a.offsets.back()!=a.nz || !std::is_sorted(a.offsets.begin(),a.offsets.end()))
    throw std::runtime_error("Invalid CSR offsets");
  for(int i=0;i<a.nz;i++) if(a.columns[i]<0 || a.columns[i]>=a.n || !std::isfinite(a.values[i]))
    throw std::runtime_error("Invalid CSR column or nonfinite value");
  if(f.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("Trailing CSR payload");
  return a;
}

template<class T> struct Context {
  int m,n,nz;
  double norm2=0;
  Device<int> offsets,columns;
  Device<T> values;
  Device<char> scratch;
  size_t scratch_size=0;
  cublasHandle_t blas=nullptr;
  cusparseHandle_t sparse=nullptr;
  cusolverDnHandle_t solver=nullptr;
  cusparseSpMatDescr_t a=nullptr;
  std::string callback_error;

  explicit Context(const Input& input):m(input.m),n(input.n),nz(input.nz),
    offsets(input.offsets.size()),columns(input.columns.size()),values(input.values.size()) {
    std::vector<T> typed(input.values.size());
    for(size_t i=0;i<typed.size();i++) { typed[i]=static_cast<T>(input.values[i]); norm2+=double(typed[i])*double(typed[i]); }
    if(!std::isfinite(norm2) || norm2==0) throw std::runtime_error("Invalid effective matrix energy");
    CUDA(cudaMemcpy(offsets.ptr,input.offsets.data(),input.offsets.size()*sizeof(int),cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(columns.ptr,input.columns.data(),input.columns.size()*sizeof(int),cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(values.ptr,typed.data(),typed.size()*sizeof(T),cudaMemcpyHostToDevice));
    BLAS(cublasCreate(&blas)); SPARSE(cusparseCreate(&sparse)); SOLVER(cusolverDnCreate(&solver));
    BLAS(cublasSetMathMode(blas,CUBLAS_PEDANTIC_MATH));
    SPARSE(cusparseCreateCsr(&a,m,n,nz,offsets.ptr,columns.ptr,values.ptr,
      CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,scalar_type<T>()));
  }
  ~Context() {
    if(a) cusparseDestroySpMat(a);
    if(solver) cusolverDnDestroy(solver);
    if(sparse) cusparseDestroy(sparse);
    if(blas) cublasDestroy(blas);
  }
  void reset_scratch() { scratch=Device<char>(); scratch_size=0; callback_error.clear(); }
  void multiply(bool transpose,const T* x,int block,int ldx,T* y,int ldy) {
    if(block<1 || ldx<(transpose?m:n) || ldy<(transpose?n:m))
      throw std::runtime_error("Invalid sparse block leading dimension");
    cusparseDnMatDescr_t dx=nullptr,dy=nullptr;
    SPARSE(cusparseCreateDnMat(&dx,transpose?m:n,block,ldx,const_cast<T*>(x),scalar_type<T>(),CUSPARSE_ORDER_COL));
    SPARSE(cusparseCreateDnMat(&dy,transpose?n:m,block,ldy,y,scalar_type<T>(),CUSPARSE_ORDER_COL));
    const T one=1,zero=0;
    size_t needed=0;
    auto op=transpose?CUSPARSE_OPERATION_TRANSPOSE:CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto rc=cusparseSpMM_bufferSize(sparse,op,CUSPARSE_OPERATION_NON_TRANSPOSE,&one,a,dx,&zero,dy,
      scalar_type<T>(),CUSPARSE_SPMM_ALG_DEFAULT,&needed);
    if(rc==CUSPARSE_STATUS_SUCCESS && needed>scratch_size) { scratch=Device<char>(needed); scratch_size=needed; }
    if(rc==CUSPARSE_STATUS_SUCCESS) rc=cusparseSpMM(sparse,op,CUSPARSE_OPERATION_NON_TRANSPOSE,
      &one,a,dx,&zero,dy,scalar_type<T>(),CUSPARSE_SPMM_ALG_DEFAULT,scratch.ptr);
    cusparseDestroyDnMat(dx); cusparseDestroyDnMat(dy);
    SPARSE(rc);
  }
};

template<class T> void gemm(Context<T>& c,cublasOperation_t opa,cublasOperation_t opb,
  int m,int n,int k,const T* a,int lda,const T* b,int ldb,T* out,int ldc) {
  const T one=1,zero=0;
  if constexpr(std::is_same<T,float>::value) BLAS(cublasSgemm(c.blas,opa,opb,m,n,k,&one,a,lda,b,ldb,&zero,out,ldc));
  else BLAS(cublasDgemm(c.blas,opa,opb,m,n,k,&one,a,lda,b,ldb,&zero,out,ldc));
}

void check_info(int* d_info,const char* operation) {
  int info=0; CUDA(cudaMemcpy(&info,d_info,sizeof(int),cudaMemcpyDeviceToHost));
  if(info) throw std::runtime_error(std::string(operation)+": devInfo="+std::to_string(info));
}

template<class T> void qr(Context<T>& c,T* matrix,int rows,int ell) {
  Device<T> tau(ell); Device<int> info(1); int first=0,second=0;
  if constexpr(std::is_same<T,float>::value) {
    SOLVER(cusolverDnSgeqrf_bufferSize(c.solver,rows,ell,matrix,rows,&first));
    SOLVER(cusolverDnSorgqr_bufferSize(c.solver,rows,ell,ell,matrix,rows,tau.ptr,&second));
  } else {
    SOLVER(cusolverDnDgeqrf_bufferSize(c.solver,rows,ell,matrix,rows,&first));
    SOLVER(cusolverDnDorgqr_bufferSize(c.solver,rows,ell,ell,matrix,rows,tau.ptr,&second));
  }
  Device<T> work(std::max(first,second));
  if constexpr(std::is_same<T,float>::value) SOLVER(cusolverDnSgeqrf(c.solver,rows,ell,matrix,rows,tau.ptr,work.ptr,first,info.ptr));
  else SOLVER(cusolverDnDgeqrf(c.solver,rows,ell,matrix,rows,tau.ptr,work.ptr,first,info.ptr));
  check_info(info.ptr,"geqrf");
  if constexpr(std::is_same<T,float>::value) SOLVER(cusolverDnSorgqr(c.solver,rows,ell,ell,matrix,rows,tau.ptr,work.ptr,second,info.ptr));
  else SOLVER(cusolverDnDorgqr(c.solver,rows,ell,ell,matrix,rows,tau.ptr,work.ptr,second,info.ptr));
  check_info(info.ptr,"orgqr");
}

template<class T> std::vector<T> spectrum(Context<T>& c,T* bt,int ell,T* thin_u=nullptr,T* full_vt=nullptr) {
  Device<T> sig(ell); Device<int> info(1); int count=0;
  if constexpr(std::is_same<T,float>::value) SOLVER(cusolverDnSgesvd_bufferSize(c.solver,c.n,ell,&count));
  else SOLVER(cusolverDnDgesvd_bufferSize(c.solver,c.n,ell,&count));
  Device<T> work(count);
  signed char left=thin_u?'S':'N',right=full_vt?'A':'N';
  if constexpr(std::is_same<T,float>::value)
    SOLVER(cusolverDnSgesvd(c.solver,left,right,c.n,ell,bt,c.n,sig.ptr,thin_u,c.n,full_vt,ell,work.ptr,count,nullptr,info.ptr));
  else SOLVER(cusolverDnDgesvd(c.solver,left,right,c.n,ell,bt,c.n,sig.ptr,thin_u,c.n,full_vt,ell,work.ptr,count,nullptr,info.ptr));
  check_info(info.ptr,"gesvd");
  std::vector<T> host(ell); CUDA(cudaMemcpy(host.data(),sig.ptr,ell*sizeof(T),cudaMemcpyDeviceToHost));
  return host;
}

struct Options {
  std::string matrix,output,json,solver="rsvd",precision="fp32";
  int k=16,ell=40,q=0,warmup=1,repeats=5,diagnostics=1,block_size=1;
  long long max_matvecs=20000;
  unsigned long long seed=20261013;
  double eps=1e-3;
};

template<class T> struct Result {
  Device<T> vectors;
  std::vector<T> sigma;
  int m,n,k;
  int converged=0,primme_rc=0,locking_issue=0,primme_min_restart_size=0;
  long long matvecs=0;
  double rho_ell=std::numeric_limits<double>::quiet_NaN();
  double rho_k=std::numeric_limits<double>::quiet_NaN();
  double pve=std::numeric_limits<double>::quiet_NaN();
  double q_orthogonality=std::numeric_limits<double>::quiet_NaN();
  double a_norm=0;
  explicit Result(int rows,int cols,int rank):vectors(size_t(rows+cols)*rank),sigma(rank),m(rows),n(cols),k(rank) {}
  T* u() { return vectors.ptr; }
  T* v() { return vectors.ptr+size_t(m)*k; }
};

template<class T> Result<T> run_rsvd(Context<T>& c,const Options& o) {
  c.reset_scratch();
  Result<T> result(c.m,c.n,o.k);
  int ell=o.ell;
  size_t gaussian_count=(size_t(c.n)*ell+1)&~size_t(1);
  Device<T> omega(gaussian_count),q(size_t(c.m)*ell),z(size_t(c.n)*ell);
  curandGenerator_t generator=nullptr;
  RNG(curandCreateGenerator(&generator,CURAND_RNG_PSEUDO_DEFAULT));
  RNG(curandSetPseudoRandomGeneratorSeed(generator,o.seed));
  if constexpr(std::is_same<T,float>::value) RNG(curandGenerateNormal(generator,omega.ptr,gaussian_count,0.F,1.F));
  else RNG(curandGenerateNormalDouble(generator,omega.ptr,gaussian_count,0.,1.));
  auto random_status=curandDestroyGenerator(generator); RNG(random_status);
  c.multiply(false,omega.ptr,ell,c.n,q.ptr,c.m); qr(c,q.ptr,c.m,ell);
  std::vector<T> previous;
  if(o.diagnostics && o.q) { c.multiply(true,q.ptr,ell,c.m,z.ptr,c.n); previous=spectrum(c,z.ptr,ell); }
  for(int step=0;step<o.q;step++) {
    c.multiply(true,q.ptr,ell,c.m,z.ptr,c.n); qr(c,z.ptr,c.n,ell);
    c.multiply(false,z.ptr,ell,c.n,q.ptr,c.m); qr(c,q.ptr,c.m,ell);
    if(o.diagnostics && step+1<o.q) { c.multiply(true,q.ptr,ell,c.m,z.ptr,c.n); previous=spectrum(c,z.ptr,ell); }
  }
  c.multiply(true,q.ptr,ell,c.m,z.ptr,c.n);
  Device<T> projected_u(size_t(c.n)*ell),projected_vt(size_t(ell)*ell);
  auto all_sigma=spectrum(c,z.ptr,ell,projected_u.ptr,projected_vt.ptr);
  gemm(c,CUBLAS_OP_N,CUBLAS_OP_T,c.m,o.k,ell,q.ptr,c.m,projected_vt.ptr,ell,result.u(),c.m);
  CUDA(cudaMemcpy(result.v(),projected_u.ptr,size_t(c.n)*o.k*sizeof(T),cudaMemcpyDeviceToDevice));
  std::copy(all_sigma.begin(),all_sigma.begin()+o.k,result.sigma.begin());
  if(o.diagnostics) {
    double captured=0,retained=0;
    for(int i=0;i<ell;i++) { double energy=double(all_sigma[i])*double(all_sigma[i]); captured+=energy; if(i<o.k) retained+=energy; }
    result.rho_ell=std::sqrt(std::max(0.,1-captured/c.norm2));
    result.rho_k=std::sqrt(std::max(0.,1-retained/c.norm2));
    if(!previous.empty()) {
      double change=0;
      for(int i=0;i<o.k;i++) change=std::max(change,std::abs(double(all_sigma[i])-double(previous[i])));
      result.pve=change/std::max(double(all_sigma[o.k-1]),1e-7);
    }
    Device<T> gram(size_t(ell)*ell); std::vector<T> host_gram(size_t(ell)*ell);
    gemm(c,CUBLAS_OP_T,CUBLAS_OP_N,ell,ell,c.m,q.ptr,c.m,q.ptr,c.m,gram.ptr,ell);
    CUDA(cudaMemcpy(host_gram.data(),gram.ptr,host_gram.size()*sizeof(T),cudaMemcpyDeviceToHost));
    double error2=0;
    for(int j=0;j<ell;j++) for(int i=0;i<ell;i++) { double delta=double(host_gram[size_t(j)*ell+i])-(i==j?1.:0.); error2+=delta*delta; }
    result.q_orthogonality=std::sqrt(error2);
  }
  result.converged=o.k;
  return result;
}

template<class T> void csr_callback(void* x,PRIMME_INT* ldx,void* y,PRIMME_INT* ldy,
  int* block,int* transpose,primme_svds_params* p,int* ierr) {
  auto* c=static_cast<Context<T>*>(p->matrix); *ierr=0;
  try {
    if(*ldx>INT32_MAX || *ldy>INT32_MAX || (*transpose!=0 && *transpose!=1))
      throw std::runtime_error("Unsupported PRIMME callback dimensions");
    if(*block>0) c->multiply(*transpose!=0,static_cast<T*>(x),*block,static_cast<int>(*ldx),static_cast<T*>(y),static_cast<int>(*ldy));
  } catch(const std::exception& e) { c->callback_error=e.what(); *ierr=1; }
}

template<class T> Result<T> run_primme(Context<T>& c,const Options& o) {
  c.reset_scratch(); Result<T> result(c.m,c.n,o.k); std::vector<T> residuals(o.k);
  primme_svds_params p; primme_svds_initialize(&p);
  p.m=p.mLocal=c.m; p.n=p.nLocal=c.n; p.numSvals=o.k; p.target=primme_svds_largest;
  p.eps=o.eps; p.matrix=&c; p.matrixMatvec=&csr_callback<T>; p.queue=&c.blas;
  p.maxBlockSize=o.block_size;
  p.maxMatvecs=o.max_matvecs; p.printLevel=0; p.outputFile=stderr; p.precondition=0;
  for(int i=0;i<4;i++) p.iseed[i]=static_cast<PRIMME_INT>((o.seed>>(12*i))&4095ULL);
  p.iseed[3]|=1;
  int method=primme_svds_set_method(primme_svds_hybrid,PRIMME_DEFAULT_METHOD,PRIMME_DEFAULT_METHOD,&p);
  if(method) { primme_svds_free(&p); throw std::runtime_error("PRIMME method setup failed"); }
  // For tiny problems upstream automatic defaults can produce a zero
  // minRestartSize (first-stage PRIMME error -18, SVDS error -118).
  // Supply a positive public-API parameter for these small dimensions.
  // Every public benchmark matrix is much larger and uses upstream defaults.
  if(std::min(c.m,c.n)<=2LL*o.block_size) {
    p.primme.minRestartSize=std::max(1,std::min(o.k,std::min(c.m,c.n)-1));
  }
  int rc;
  if constexpr(std::is_same<T,float>::value) rc=cublas_sprimme_svds(result.sigma.data(),result.vectors.ptr,residuals.data(),&p);
  else rc=cublas_dprimme_svds(result.sigma.data(),result.vectors.ptr,residuals.data(),&p);
  result.primme_rc=rc; result.converged=p.initSize; result.matvecs=p.stats.numMatvecs; result.a_norm=p.aNorm;
  result.locking_issue=p.stats.lockingIssue;
  result.primme_min_restart_size=p.primme.minRestartSize;
  primme_svds_free(&p);
  if(rc || result.converged!=o.k) throw std::runtime_error("PRIMME rc="+std::to_string(rc)+", converged="+
    std::to_string(result.converged)+"/"+std::to_string(o.k)+"; "+c.callback_error);
  return result;
}

template<class T> void write_result(const std::string& path,Result<T>& r) {
  std::vector<T> host(size_t(r.m+r.n)*r.k);
  CUDA(cudaMemcpy(host.data(),r.vectors.ptr,host.size()*sizeof(T),cudaMemcpyDeviceToHost));
  std::ofstream f(path,std::ios::binary); uint64_t shape[3]={uint64_t(r.m),uint64_t(r.n),uint64_t(r.k)};
  uint32_t kind[2]={sizeof(T),0};
  f.write("PGFACT01",8); f.write(reinterpret_cast<char*>(shape),24); f.write(reinterpret_cast<char*>(kind),8);
  f.write(reinterpret_cast<char*>(r.sigma.data()),r.sigma.size()*sizeof(T));
  f.write(reinterpret_cast<char*>(host.data()),host.size()*sizeof(T));
  if(!f) throw std::runtime_error("Failed to save complete singular factors");
}

void number(std::ostream& f,double x) { if(std::isfinite(x)) f<<x; else f<<"null"; }

template<class T> int execute(const Options& o) {
  auto input_start=Clock::now(); Input a=load_input(o.matrix); double input_ms=elapsed(input_start);
  if(o.k<1 || o.k>std::min(a.m,a.n) || (o.solver=="rsvd" && (o.ell<o.k || o.ell>std::min(a.m,a.n))) ||
     o.q<0 || o.q>8 || o.warmup<0 || o.repeats<1 || !(o.eps>0 && o.eps<1) || o.max_matvecs<1 ||
     o.block_size<1 || o.block_size>std::min(a.m,a.n))
    throw std::runtime_error("Invalid rank, width, power, tolerance, repetition count, or block size");
  auto setup_start=Clock::now(); Context<T> context(a); CUDA(cudaDeviceSynchronize()); double setup_ms=elapsed(setup_start);
  int warmup_locking_issues=0;
  for(int i=0;i<o.warmup;i++) {
    auto warm=o.solver=="rsvd"?run_rsvd(context,o):run_primme(context,o);
    CUDA(cudaDeviceSynchronize()); warmup_locking_issues+=warm.locking_issue!=0;
  }
  std::vector<double> times,export_times;
  std::vector<long long> matvec_samples;
  std::vector<int> locking_samples;
  Result<T> final(a.m,a.n,o.k);
  for(int rep=0;rep<o.repeats;rep++) {
    CUDA(cudaDeviceSynchronize()); auto t=Clock::now();
    auto r=o.solver=="rsvd"?run_rsvd(context,o):run_primme(context,o);
    CUDA(cudaDeviceSynchronize()); times.push_back(elapsed(t));
    matvec_samples.push_back(r.matvecs); locking_samples.push_back(r.locking_issue);
    auto export_start=Clock::now();
    write_result(o.output+".repeat_"+std::to_string(rep)+".bin",r);
    export_times.push_back(elapsed(export_start));
    final=std::move(r);
  }
  auto output_start=Clock::now(); write_result(o.output,final); double output_ms=elapsed(output_start);
  auto sorted=times; std::sort(sorted.begin(),sorted.end()); double median=sorted[sorted.size()/2];
  if(sorted.size()%2==0) median=(sorted[sorted.size()/2-1]+median)/2;
  cudaDeviceProp device{}; CUDA(cudaGetDeviceProperties(&device,0)); int driver=0,runtime=0;
  CUDA(cudaDriverGetVersion(&driver)); CUDA(cudaRuntimeGetVersion(&runtime));
  std::ofstream f(o.json); f<<std::setprecision(17);
  f<<"{\n\"status\":\"ok\",\"solver\":\""<<(o.solver=="primme"?"primme_svds_cuda":(o.diagnostics?"rsvd_diagnostic_full":"rsvd_full"))
   <<"\",\"precision\":\""<<o.precision<<"\",\"rows\":"<<a.m<<",\"cols\":"<<a.n<<",\"nnz\":"<<a.nz
   <<",\"rank\":"<<o.k<<",\"ell\":"<<(o.solver=="rsvd"?o.ell:0)<<",\"power\":"<<(o.solver=="rsvd"?o.q:0)
   <<",\"seed\":"<<o.seed<<",\"eps\":"<<o.eps<<",\"block_size\":"<<(o.solver=="primme"?o.block_size:0)
   <<",\"warmup\":"<<o.warmup<<",\"repeats\":"<<o.repeats
   <<",\"input_read_ms\":"<<input_ms<<",\"resident_setup_ms\":"<<setup_ms<<",\"compute_ms\":"<<median
   <<",\"output_copy_and_write_ms\":"<<output_ms<<",\"compute_samples_ms\":[";
  for(size_t i=0;i<times.size();i++) { if(i) f<<','; f<<times[i]; }
  f<<"],\"repeat_factor_export_samples_ms\":[";
  for(size_t i=0;i<export_times.size();i++) { if(i) f<<','; f<<export_times[i]; }
  f<<"],\"converged_rank\":"<<final.converged<<",\"matvecs\":"<<final.matvecs<<",\"primme_rc\":"<<final.primme_rc
   <<",\"primme_locking_issue\":"<<std::any_of(locking_samples.begin(),locking_samples.end(),[](int v){return v!=0;})
   <<",\"warmup_primme_locking_issue_count\":"<<warmup_locking_issues
   <<",\"primme_min_restart_size\":"<<final.primme_min_restart_size
   <<",\"matvec_samples\":[";
  for(size_t i=0;i<matvec_samples.size();i++) { if(i) f<<','; f<<matvec_samples[i]; }
  f<<"],\"primme_locking_issue_samples\":[";
  for(size_t i=0;i<locking_samples.size();i++) { if(i) f<<','; f<<locking_samples[i]; }
  f<<"],\"primme_a_norm\":"<<final.a_norm<<",\"rho_ell_energy\":"; number(f,final.rho_ell);
  f<<",\"rho_k_energy\":"; number(f,final.rho_k); f<<",\"pve\":"; number(f,final.pve);
  f<<",\"q_orthogonality\":"; number(f,final.q_orthogonality);
  f<<",\"gpu\":\""<<device.name<<"\",\"compute_capability\":\""<<device.major<<device.minor
   <<"\",\"driver_api\":"<<driver<<",\"cuda_runtime\":"<<runtime<<"}\n";
  if(!f) throw std::runtime_error("Failed to save solver record");
  std::printf("%s rank=%d median_ms=%.6f complete_factors=1\n",o.solver.c_str(),o.k,median);
  return 0;
}

int main(int argc,char** argv) {
  try {
    Options o;
    if(argc==2 && std::string(argv[1])=="--help") {
      std::puts("gpu_compare --matrix input.csr --solver rsvd|primme --rank k --precision fp32|fp64 --output factors.bin --json record.json [--ell L --power q --eps tol --block-size n --seed n --warmup n --repeats n --diagnostics 0|1 --max-matvecs n]"); return 0;
    }
    for(int i=1;i<argc;i+=2) {
      if(i+1>=argc) throw std::runtime_error("Every argument needs a value");
      std::string key=argv[i],value=argv[i+1];
      if(key=="--matrix") o.matrix=value; else if(key=="--solver") o.solver=value;
      else if(key=="--output") o.output=value; else if(key=="--json") o.json=value;
      else if(key=="--precision") o.precision=value; else if(key=="--rank") o.k=std::stoi(value);
      else if(key=="--ell") o.ell=std::stoi(value); else if(key=="--power") o.q=std::stoi(value);
      else if(key=="--eps") o.eps=std::stod(value); else if(key=="--seed") o.seed=std::stoull(value);
      else if(key=="--warmup") o.warmup=std::stoi(value); else if(key=="--repeats") o.repeats=std::stoi(value);
      else if(key=="--diagnostics") o.diagnostics=std::stoi(value); else if(key=="--max-matvecs") o.max_matvecs=std::stoll(value);
      else if(key=="--block-size") o.block_size=std::stoi(value);
      else throw std::runtime_error("Unknown option: "+key);
    }
    if(o.matrix.empty() || o.output.empty() || o.json.empty() || (o.solver!="primme" && o.solver!="rsvd") ||
       (o.precision!="fp32" && o.precision!="fp64") || (o.diagnostics!=0 && o.diagnostics!=1))
      throw std::runtime_error("Required input/output, solver, or precision is invalid; see --help");
    return o.precision=="fp32"?execute<float>(o):execute<double>(o);
  } catch(const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
