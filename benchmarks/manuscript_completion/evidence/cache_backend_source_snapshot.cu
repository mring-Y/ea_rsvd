// Measure GPU reuse of a resident randomized-SVD state.
// The cache contains Q, the SVD factors of B^T=A^T Q, and singular values.
// It is intentionally independent of the published PRIMME comparison binary.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <curand.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA(x) do { auto e=(x); if(e!=cudaSuccess) throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(e)); } while(0)
#define BLAS(x) do { auto e=(x); if(e!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(#x)+": "+std::to_string(e)); } while(0)
#define SPARSE(x) do { auto e=(x); if(e!=CUSPARSE_STATUS_SUCCESS) throw std::runtime_error(std::string(#x)+": "+std::to_string(e)); } while(0)
#define SOLVER(x) do { auto e=(x); if(e!=CUSOLVER_STATUS_SUCCESS) throw std::runtime_error(std::string(#x)+": "+std::to_string(e)); } while(0)
#define RNG(x) do { auto e=(x); if(e!=CURAND_STATUS_SUCCESS) throw std::runtime_error(std::string(#x)+": "+std::to_string(e)); } while(0)

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }

template<class T> struct Device {
  T* ptr=nullptr;
  Device()=default;
  explicit Device(size_t n) { if(n) CUDA(cudaMalloc(reinterpret_cast<void**>(&ptr),n*sizeof(T))); }
  ~Device() { if(ptr) cudaFree(ptr); }
  Device(const Device&)=delete; Device& operator=(const Device&)=delete;
  Device(Device&& x) noexcept:ptr(x.ptr) { x.ptr=nullptr; }
  Device& operator=(Device&& x) noexcept { if(ptr) cudaFree(ptr); ptr=x.ptr; x.ptr=nullptr; return *this; }
};

struct Input { int m=0,n=0,nz=0; std::vector<int> offsets,columns; std::vector<double> values; };
template<class T> static void read_bytes(std::ifstream& f,T* p,size_t n) {
  f.read(reinterpret_cast<char*>(p),static_cast<std::streamsize>(n*sizeof(T)));
  if(!f) throw std::runtime_error("truncated CSR input");
}
static Input load_input(const std::string& path) {
  std::ifstream f(path,std::ios::binary); if(!f) throw std::runtime_error("cannot open CSR input");
  char magic[8]; uint64_t shape[3]; read_bytes(f,magic,8); read_bytes(f,shape,3);
  if(std::memcmp(magic,"PGCSR001",8) || !shape[0] || !shape[1] || !shape[2] || shape[0]>INT32_MAX || shape[1]>INT32_MAX || shape[2]>INT32_MAX)
    throw std::runtime_error("invalid CSR header");
  Input a; a.m=(int)shape[0]; a.n=(int)shape[1]; a.nz=(int)shape[2];
  a.offsets.resize(a.m+1); a.columns.resize(a.nz); a.values.resize(a.nz);
  read_bytes(f,a.offsets.data(),a.offsets.size()); read_bytes(f,a.columns.data(),a.columns.size()); read_bytes(f,a.values.data(),a.values.size());
  if(a.offsets.front()!=0 || a.offsets.back()!=a.nz || !std::is_sorted(a.offsets.begin(),a.offsets.end())) throw std::runtime_error("invalid CSR offsets");
  for(int j:a.columns) if(j<0 || j>=a.n) throw std::runtime_error("invalid CSR column");
  for(double x:a.values) if(!std::isfinite(x)) throw std::runtime_error("nonfinite CSR value");
  return a;
}

struct Context {
  int m,n,nz; double norm2=0;
  Device<int> offsets,columns; Device<float> values,scratch; size_t scratch_size=0;
  cublasHandle_t blas=nullptr; cusparseHandle_t sparse=nullptr; cusolverDnHandle_t solver=nullptr; cusparseSpMatDescr_t a=nullptr;
  explicit Context(const Input& in):m(in.m),n(in.n),nz(in.nz),offsets(in.offsets.size()),columns(in.columns.size()),values(in.values.size()) {
    std::vector<float> typed(in.nz);
    for(int i=0;i<in.nz;i++) { typed[i]=(float)in.values[i]; norm2+=(double)typed[i]*typed[i]; }
    CUDA(cudaMemcpy(offsets.ptr,in.offsets.data(),in.offsets.size()*sizeof(int),cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(columns.ptr,in.columns.data(),in.columns.size()*sizeof(int),cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(values.ptr,typed.data(),typed.size()*sizeof(float),cudaMemcpyHostToDevice));
    BLAS(cublasCreate(&blas)); SPARSE(cusparseCreate(&sparse)); SOLVER(cusolverDnCreate(&solver));
    BLAS(cublasSetMathMode(blas,CUBLAS_PEDANTIC_MATH));
    SPARSE(cusparseCreateCsr(&a,m,n,nz,offsets.ptr,columns.ptr,values.ptr,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));
  }
  ~Context() { if(a) cusparseDestroySpMat(a); if(solver) cusolverDnDestroy(solver); if(sparse) cusparseDestroy(sparse); if(blas) cublasDestroy(blas); }
  void multiply(bool transpose,const float* x,int block,int ldx,float* y,int ldy) {
    cusparseDnMatDescr_t dx=nullptr,dy=nullptr; auto op=transpose?CUSPARSE_OPERATION_TRANSPOSE:CUSPARSE_OPERATION_NON_TRANSPOSE;
    SPARSE(cusparseCreateDnMat(&dx,transpose?m:n,block,ldx,const_cast<float*>(x),CUDA_R_32F,CUSPARSE_ORDER_COL));
    SPARSE(cusparseCreateDnMat(&dy,transpose?n:m,block,ldy,y,CUDA_R_32F,CUSPARSE_ORDER_COL));
    const float one=1,zero=0; size_t needed=0;
    auto status=cusparseSpMM_bufferSize(sparse,op,CUSPARSE_OPERATION_NON_TRANSPOSE,&one,a,dx,&zero,dy,CUDA_R_32F,CUSPARSE_SPMM_ALG_DEFAULT,&needed);
    if(status==CUSPARSE_STATUS_SUCCESS && needed>scratch_size) { scratch=Device<float>((needed+sizeof(float)-1)/sizeof(float)); scratch_size=needed; }
    if(status==CUSPARSE_STATUS_SUCCESS) status=cusparseSpMM(sparse,op,CUSPARSE_OPERATION_NON_TRANSPOSE,&one,a,dx,&zero,dy,CUDA_R_32F,CUSPARSE_SPMM_ALG_DEFAULT,scratch.ptr);
    cusparseDestroyDnMat(dx); cusparseDestroyDnMat(dy); SPARSE(status);
  }
  void gemm(cublasOperation_t oa,cublasOperation_t ob,int m0,int n0,int k,const float* x,int ldx,const float* y,int ldy,float* z,int ldz) {
    const float one=1,zero=0; BLAS(cublasSgemm(blas,oa,ob,m0,n0,k,&one,x,ldx,y,ldy,&zero,z,ldz));
  }
};

static void qr(Context& c,float* x,int rows,int ell) {
  Device<float> tau(ell); Device<int> info(1); int qrf=0,org=0;
  SOLVER(cusolverDnSgeqrf_bufferSize(c.solver,rows,ell,x,rows,&qrf));
  SOLVER(cusolverDnSorgqr_bufferSize(c.solver,rows,ell,ell,x,rows,tau.ptr,&org));
  Device<float> work(std::max(qrf,org));
  SOLVER(cusolverDnSgeqrf(c.solver,rows,ell,x,rows,tau.ptr,work.ptr,qrf,info.ptr));
  int h=0; CUDA(cudaMemcpy(&h,info.ptr,sizeof(int),cudaMemcpyDeviceToHost)); if(h) throw std::runtime_error("geqrf devInfo="+std::to_string(h));
  SOLVER(cusolverDnSorgqr(c.solver,rows,ell,ell,x,rows,tau.ptr,work.ptr,org,info.ptr));
  CUDA(cudaMemcpy(&h,info.ptr,sizeof(int),cudaMemcpyDeviceToHost)); if(h) throw std::runtime_error("orgqr devInfo="+std::to_string(h));
}

struct Cache {
  Device<float> q,ub,vt,sigma;
  int m,n,ell; std::vector<float> host_sigma;
  Cache(int rows,int cols,int width):q((size_t)rows*width),ub((size_t)cols*width),vt((size_t)width*width),sigma(width),m(rows),n(cols),ell(width),host_sigma(width) {}
};

static Cache build_cache(Context& c,int ell,int power,unsigned long long seed) {
  size_t random_count=(size_t)c.n*ell; random_count+=(random_count%2);
  Cache cache(c.m,c.n,ell); Device<float> omega(random_count),z((size_t)c.n*ell);
  curandGenerator_t rng=nullptr; RNG(curandCreateGenerator(&rng,CURAND_RNG_PSEUDO_DEFAULT)); RNG(curandSetPseudoRandomGeneratorSeed(rng,seed));
  RNG(curandGenerateNormal(rng,omega.ptr,random_count,0.0F,1.0F)); RNG(curandDestroyGenerator(rng));
  c.multiply(false,omega.ptr,ell,c.n,cache.q.ptr,c.m); qr(c,cache.q.ptr,c.m,ell);
  for(int step=0;step<power;step++) { c.multiply(true,cache.q.ptr,ell,c.m,z.ptr,c.n); qr(c,z.ptr,c.n,ell); c.multiply(false,z.ptr,ell,c.n,cache.q.ptr,c.m); qr(c,cache.q.ptr,c.m,ell); }
  c.multiply(true,cache.q.ptr,ell,c.m,z.ptr,c.n); CUDA(cudaMemcpy(cache.ub.ptr,z.ptr,(size_t)c.n*ell*sizeof(float),cudaMemcpyDeviceToDevice));
  int work_size=0; Device<int> info(1); SOLVER(cusolverDnSgesvd_bufferSize(c.solver,c.n,ell,&work_size)); Device<float> work(work_size);
  SOLVER(cusolverDnSgesvd(c.solver,'S','S',c.n,ell,z.ptr,c.n,cache.sigma.ptr,cache.ub.ptr,c.n,cache.vt.ptr,ell,work.ptr,work_size,nullptr,info.ptr));
  int h=0; CUDA(cudaMemcpy(&h,info.ptr,sizeof(int),cudaMemcpyDeviceToHost)); if(h) throw std::runtime_error("SVD devInfo="+std::to_string(h));
  CUDA(cudaMemcpy(cache.host_sigma.data(),cache.sigma.ptr,ell*sizeof(float),cudaMemcpyDeviceToHost));
  return cache;
}

static void form_factors(Context& c,const Cache& cache,int k,Device<float>& u,Device<float>& v) {
  u=Device<float>((size_t)c.m*k); v=Device<float>((size_t)c.n*k);
  c.gemm(CUBLAS_OP_N,CUBLAS_OP_T,c.m,k,cache.ell,cache.q.ptr,c.m,cache.vt.ptr,cache.ell,u.ptr,c.m);
  for(int j=0;j<k;j++) {
    CUDA(cudaMemcpy(v.ptr+(size_t)j*c.n,cache.ub.ptr+(size_t)j*c.n,(size_t)c.n*sizeof(float),cudaMemcpyDeviceToDevice));
  }
}

static void write_factors(const std::string& path,Context& c,Device<float>& u,Device<float>& v,const std::vector<float>& sigma,int k) {
  std::vector<float> hu((size_t)c.m*k),hv((size_t)c.n*k); CUDA(cudaMemcpy(hu.data(),u.ptr,hu.size()*sizeof(float),cudaMemcpyDeviceToHost)); CUDA(cudaMemcpy(hv.data(),v.ptr,hv.size()*sizeof(float),cudaMemcpyDeviceToHost));
  std::ofstream f(path,std::ios::binary); uint64_t shape[3]={uint64_t(c.m),uint64_t(c.n),uint64_t(k)}; uint32_t kind[2]={4,0};
  if(!f) throw std::runtime_error("cannot open factor output");
  f.write("PGFACT01",8); f.write(reinterpret_cast<char*>(shape),24); f.write(reinterpret_cast<char*>(kind),8); f.write(reinterpret_cast<const char*>(sigma.data()),k*sizeof(float)); f.write(reinterpret_cast<char*>(hu.data()),hu.size()*sizeof(float)); f.write(reinterpret_cast<char*>(hv.data()),hv.size()*sizeof(float));
  if(!f) throw std::runtime_error("cannot write factor output");
}

static double median(std::vector<double> values) {
  if(values.empty()) throw std::runtime_error("no timing samples");
  std::sort(values.begin(),values.end()); size_t middle=values.size()/2;
  return values.size()%2?values[middle]:0.5*(values[middle-1]+values[middle]);
}

int main(int argc,char** argv) {
  try {
    if(argc!=8) { std::cerr<<"usage: cached_reuse matrix.csr output.csv ell power seed repeats factor_dir\n"; return 64; }
    std::string matrix=argv[1],csv=argv[2],factor_dir=argv[7]; int ell=std::stoi(argv[3]),power=std::stoi(argv[4]),repeats=std::stoi(argv[6]); unsigned long long seed=std::stoull(argv[5]);
    if(ell<1 || power<0 || power>8 || repeats<1) throw std::runtime_error("invalid options");
    Input input=load_input(matrix); if(ell>std::min(input.m,input.n)) throw std::runtime_error("width exceeds smaller matrix dimension");
    Context c(input); CUDA(cudaDeviceSynchronize());
    std::ofstream out(csv),samples(csv+".samples.csv"); if(!out || !samples) throw std::runtime_error("cannot open timing output");
    samples<<"operation,rank,sample,ms\n"<<std::setprecision(17);
    double cold_ms=0;
    { auto start=Clock::now(); Cache cold=build_cache(c,ell,power,seed); CUDA(cudaDeviceSynchronize()); cold_ms=ms_since(start); }
    std::vector<double> build_times;
    for(int rep=0;rep<repeats;rep++) { CUDA(cudaDeviceSynchronize()); auto t=Clock::now(); Cache warm=build_cache(c,ell,power,seed); CUDA(cudaDeviceSynchronize()); double ms=ms_since(t); build_times.push_back(ms); samples<<"cache_build,0,"<<rep<<','<<ms<<'\n'; }
    double build_ms=median(build_times); Cache cache=build_cache(c,ell,power,seed); CUDA(cudaDeviceSynchronize());
    out<<"matrix,rows,cols,ell,power,seed,rank,cache_build_ms,cold_cache_build_ms,cached_spectrum_select_ms,cached_factor_ms,independent_recompute_ms,repeats,warmups,sigma_k\n"; out<<std::setprecision(17);
    for(int k: {8,16,32}) { if(k>ell) continue; Device<float> u,v; form_factors(c,cache,k,u,v); CUDA(cudaDeviceSynchronize());
      std::vector<double> select_times,form_times,recompute_times;
      for(int rep=0;rep<repeats;rep++) {
        CUDA(cudaDeviceSynchronize()); auto t=Clock::now(); float selected=cache.host_sigma[k-1]; (void)selected; CUDA(cudaMemcpy(&selected,cache.sigma.ptr+ k-1,sizeof(float),cudaMemcpyDeviceToHost)); CUDA(cudaDeviceSynchronize()); select_times.push_back(ms_since(t));
        CUDA(cudaDeviceSynchronize()); t=Clock::now(); Device<float> cu,cv; form_factors(c,cache,k,cu,cv); CUDA(cudaDeviceSynchronize()); form_times.push_back(ms_since(t));
        CUDA(cudaDeviceSynchronize()); t=Clock::now(); Cache fresh=build_cache(c,ell,power,seed); Device<float> iu,iv; form_factors(c,fresh,k,iu,iv); CUDA(cudaDeviceSynchronize()); recompute_times.push_back(ms_since(t));
        samples<<"spectrum_lookup,"<<k<<','<<rep<<','<<select_times.back()<<'\n';
        samples<<"cached_factor,"<<k<<','<<rep<<','<<form_times.back()<<'\n';
        samples<<"independent_recompute,"<<k<<','<<rep<<','<<recompute_times.back()<<'\n';
      }
      double select=median(select_times), form=median(form_times), recompute=median(recompute_times);
      std::string prefix=factor_dir+"/"+std::to_string(k); write_factors(prefix+"_cached.bin",c,u,v,std::vector<float>(cache.host_sigma.begin(),cache.host_sigma.begin()+k),k);
      Cache fresh=build_cache(c,ell,power,seed); Device<float> iu,iv; form_factors(c,fresh,k,iu,iv); write_factors(prefix+"_independent.bin",c,iu,iv,std::vector<float>(fresh.host_sigma.begin(),fresh.host_sigma.begin()+k),k);
      out<<std::filesystem::path(matrix).stem().string()<<','<<c.m<<','<<c.n<<','<<ell<<','<<power<<','<<seed<<','<<k<<','<<build_ms<<','<<cold_ms<<','<<select<<','<<form<<','<<recompute<<','<<repeats<<",1,"<<cache.host_sigma[k-1]<<'\n';
      std::cout<<"rank="<<k<<" factor_ms="<<form<<" recompute_ms="<<recompute<<std::endl;
    }
    std::cout<<"cached_reuse matrix="<<matrix<<" build_ms="<<build_ms<<"\n"; return 0;
  } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<"\n"; return 1; }
}
