#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

struct CsrMatrix { int rows = 0, cols = 0; std::vector<int> offsets, columns; std::vector<float> values; };

static bool read_mtx(const std::string& path, CsrMatrix* out) {
  std::ifstream in(path); std::string line;
  if (!in || !std::getline(in, line) || line.rfind("%%MatrixMarket matrix coordinate", 0) != 0) return false;
  const bool pattern = line.find("pattern") != std::string::npos;
  do { if (!std::getline(in, line)) return false; } while (line.empty() || line[0] == '%');
  int nnz = 0; std::istringstream shape(line); if (!(shape >> out->rows >> out->cols >> nnz)) return false;
  std::vector<std::vector<std::pair<int, float>>> rows(out->rows);
  for (int i = 0; i < nnz; ++i) {
    int r = 0, c = 0; float v = 1.0F;
    if (!(in >> r >> c) || r < 1 || r > out->rows || c < 1 || c > out->cols) return false;
    if (!pattern && !(in >> v)) return false;
    rows[r - 1].push_back({c - 1, v});
  }
  out->offsets.assign(out->rows + 1, 0); out->columns.reserve(nnz); out->values.reserve(nnz);
  for (int r = 0; r < out->rows; ++r) {
    out->offsets[r + 1] = out->offsets[r] + static_cast<int>(rows[r].size());
    for (const auto& x : rows[r]) { out->columns.push_back(x.first); out->values.push_back(x.second); }
  }
  return true;
}

static CsrMatrix synthetic(int rows, int cols, int entries_per_row) {
  CsrMatrix a; a.rows = rows; a.cols = cols; a.offsets.assign(rows + 1, 0);
  std::mt19937 gen(20260928); std::uniform_int_distribution<int> col(0, cols - 1); std::uniform_real_distribution<float> val(-1.0F, 1.0F);
  for (int r = 0; r < rows; ++r) {
    std::vector<int> chosen; chosen.reserve(entries_per_row);
    while (static_cast<int>(chosen.size()) < entries_per_row) { int c = col(gen); if (std::find(chosen.begin(), chosen.end(), c) == chosen.end()) chosen.push_back(c); }
    std::sort(chosen.begin(), chosen.end());
    for (int c : chosen) { a.columns.push_back(c); a.values.push_back(val(gen)); }
    a.offsets[r + 1] = static_cast<int>(a.columns.size());
  }
  return a;
}

static const char* cuda_name(cudaError_t e) { return cudaGetErrorString(e); }
static bool check_cuda(cudaError_t e, const char* file, int line) { if (e == cudaSuccess) return true; std::fprintf(stderr, "CUDA %s:%d: %s\n", file, line, cuda_name(e)); return false; }
static bool check_sparse(cusparseStatus_t e, const char* file, int line) { if (e == CUSPARSE_STATUS_SUCCESS) return true; std::fprintf(stderr, "cuSPARSE %s:%d: %d\n", file, line, static_cast<int>(e)); return false; }
static bool check_solver(cusolverStatus_t e, const char* file, int line) { if (e == CUSOLVER_STATUS_SUCCESS) return true; std::fprintf(stderr, "cuSOLVER %s:%d: %d\n", file, line, static_cast<int>(e)); return false; }
static bool check_blas(cublasStatus_t e, const char* file, int line) { if (e == CUBLAS_STATUS_SUCCESS) return true; std::fprintf(stderr, "cuBLAS %s:%d: %d\n", file, line, static_cast<int>(e)); return false; }
#define CUD(x) do { if (!check_cuda((x), __FILE__, __LINE__)) return false; } while (0)
#define SP(x) do { if (!check_sparse((x), __FILE__, __LINE__)) return false; } while (0)
#define SOL(x) do { if (!check_solver((x), __FILE__, __LINE__)) return false; } while (0)
#define BLAS(x) do { if (!check_blas((x), __FILE__, __LINE__)) return false; } while (0)

__global__ static void fill_omega(float* dst, int n, int l, unsigned seed) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = static_cast<size_t>(n) * l;
  if (i >= total) return;
  unsigned x = static_cast<unsigned>(i) ^ seed;
  x = 1664525u * x + 1013904223u; const float u = (static_cast<float>(x & 0x00ffffffu) + 1.0F) / 16777217.0F;
  x = 1664525u * x + 1013904223u; const float v = (static_cast<float>(x & 0x00ffffffu) + 1.0F) / 16777217.0F;
  dst[i] = sqrtf(-2.0F * logf(u)) * cosf(6.28318530718F * v);
}

struct Lane {
  cudaStream_t stream = nullptr; cusparseHandle_t sparse = nullptr; cusolverDnHandle_t solver = nullptr; cublasHandle_t blas = nullptr;
  cusparseSpMatDescr_t mat = nullptr; cusparseDnMatDescr_t omega = nullptr, y = nullptr, b = nullptr;
  float* d_omega = nullptr; float* d_y = nullptr; float* d_b = nullptr; float* d_b_check = nullptr; float* d_tau = nullptr; float* d_qtq = nullptr; float* d_sigma = nullptr; float* d_qr_work = nullptr; float* d_svd_work = nullptr; int* d_info = nullptr; void* d_spmm = nullptr;
  // cuSOLVER can request different workspaces for geqrf and orgqr.
  int qr_geqrf_work = 0, qr_orgqr_work = 0, qr_work = 0, svd_work = 0; size_t spmm_bytes = 0;
  cudaEvent_t run_start = nullptr, q_start = nullptr, q_end = nullptr, qr_start = nullptr, qr_end = nullptr, b_start = nullptr, b_end = nullptr, svd_start = nullptr, svd_end = nullptr;
};

static void destroy_lane(Lane* x) {
  if (x->run_start) cudaEventDestroy(x->run_start); if (x->q_start) cudaEventDestroy(x->q_start); if (x->q_end) cudaEventDestroy(x->q_end); if (x->qr_start) cudaEventDestroy(x->qr_start); if (x->qr_end) cudaEventDestroy(x->qr_end); if (x->b_start) cudaEventDestroy(x->b_start); if (x->b_end) cudaEventDestroy(x->b_end); if (x->svd_start) cudaEventDestroy(x->svd_start); if (x->svd_end) cudaEventDestroy(x->svd_end);
  if (x->mat) cusparseDestroySpMat(x->mat); if (x->omega) cusparseDestroyDnMat(x->omega); if (x->y) cusparseDestroyDnMat(x->y); if (x->b) cusparseDestroyDnMat(x->b);
  if (x->sparse) cusparseDestroy(x->sparse); if (x->solver) cusolverDnDestroy(x->solver); if (x->blas) cublasDestroy(x->blas); if (x->stream) cudaStreamDestroy(x->stream);
  cudaFree(x->d_spmm); cudaFree(x->d_svd_work); cudaFree(x->d_qr_work); cudaFree(x->d_info); cudaFree(x->d_sigma); cudaFree(x->d_qtq); cudaFree(x->d_tau); cudaFree(x->d_b_check); cudaFree(x->d_b); cudaFree(x->d_y); cudaFree(x->d_omega); *x = {};
}

static bool init_lane(Lane* x, const CsrMatrix& a, int l, const int* d_offsets, const int* d_columns, const float* d_values) {
  CUD(cudaStreamCreateWithFlags(&x->stream, cudaStreamNonBlocking)); SP(cusparseCreate(&x->sparse)); SOL(cusolverDnCreate(&x->solver)); BLAS(cublasCreate(&x->blas));
  SP(cusparseSetStream(x->sparse, x->stream)); SOL(cusolverDnSetStream(x->solver, x->stream)); BLAS(cublasSetStream(x->blas, x->stream));
  CUD(cudaEventCreate(&x->run_start)); CUD(cudaEventCreate(&x->q_start)); CUD(cudaEventCreate(&x->q_end)); CUD(cudaEventCreate(&x->qr_start)); CUD(cudaEventCreate(&x->qr_end)); CUD(cudaEventCreate(&x->b_start)); CUD(cudaEventCreate(&x->b_end)); CUD(cudaEventCreate(&x->svd_start)); CUD(cudaEventCreate(&x->svd_end));
  CUD(cudaMalloc(&x->d_omega, sizeof(float) * static_cast<size_t>(a.cols) * l)); CUD(cudaMalloc(&x->d_y, sizeof(float) * static_cast<size_t>(a.rows) * l)); CUD(cudaMalloc(&x->d_b, sizeof(float) * static_cast<size_t>(a.cols) * l)); CUD(cudaMalloc(&x->d_b_check, sizeof(float) * static_cast<size_t>(a.cols) * l)); CUD(cudaMalloc(&x->d_tau, sizeof(float) * l)); CUD(cudaMalloc(&x->d_qtq, sizeof(float) * l * l)); CUD(cudaMalloc(&x->d_sigma, sizeof(float) * l)); CUD(cudaMalloc(&x->d_info, sizeof(int)));
  SP(cusparseCreateCsr(&x->mat, a.rows, a.cols, static_cast<int>(a.values.size()), const_cast<int*>(d_offsets), const_cast<int*>(d_columns), const_cast<float*>(d_values), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  SP(cusparseCreateDnMat(&x->omega, a.cols, l, a.cols, x->d_omega, CUDA_R_32F, CUSPARSE_ORDER_COL)); SP(cusparseCreateDnMat(&x->y, a.rows, l, a.rows, x->d_y, CUDA_R_32F, CUSPARSE_ORDER_COL)); SP(cusparseCreateDnMat(&x->b, a.cols, l, a.cols, x->d_b, CUDA_R_32F, CUSPARSE_ORDER_COL));
  const float one = 1.0F, zero = 0.0F; SP(cusparseSpMM_bufferSize(x->sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, x->mat, x->omega, &zero, x->y, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &x->spmm_bytes)); CUD(cudaMalloc(&x->d_spmm, x->spmm_bytes));
  SOL(cusolverDnSgeqrf_bufferSize(x->solver, a.rows, l, x->d_y, a.rows, &x->qr_geqrf_work)); SOL(cusolverDnSorgqr_bufferSize(x->solver, a.rows, l, l, x->d_y, a.rows, x->d_tau, &x->qr_orgqr_work)); x->qr_work = std::max(x->qr_geqrf_work, x->qr_orgqr_work); CUD(cudaMalloc(&x->d_qr_work, sizeof(float) * static_cast<size_t>(x->qr_work)));
  SOL(cusolverDnSgesvd_bufferSize(x->solver, a.cols, l, &x->svd_work)); CUD(cudaMalloc(&x->d_svd_work, sizeof(float) * x->svd_work));
  return true;
}

static bool enqueue_lane(Lane* x, const CsrMatrix& a, int l, unsigned seed, bool preserve_b) {
  const float one = 1.0F, zero = 0.0F; const int threads = 256; const int blocks = static_cast<int>((static_cast<size_t>(a.cols) * l + threads - 1) / threads);
  CUD(cudaEventRecord(x->run_start, x->stream)); fill_omega<<<blocks, threads, 0, x->stream>>>(x->d_omega, a.cols, l, seed); CUD(cudaGetLastError());
  CUD(cudaEventRecord(x->q_start, x->stream)); SP(cusparseSpMM(x->sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, x->mat, x->omega, &zero, x->y, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, x->d_spmm)); CUD(cudaEventRecord(x->q_end, x->stream));
  CUD(cudaEventRecord(x->qr_start, x->stream)); SOL(cusolverDnSgeqrf(x->solver, a.rows, l, x->d_y, a.rows, x->d_tau, x->d_qr_work, x->qr_geqrf_work, x->d_info)); SOL(cusolverDnSorgqr(x->solver, a.rows, l, l, x->d_y, a.rows, x->d_tau, x->d_qr_work, x->qr_orgqr_work, x->d_info)); CUD(cudaEventRecord(x->qr_end, x->stream));
  CUD(cudaEventRecord(x->b_start, x->stream)); SP(cusparseSpMM(x->sparse, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, x->mat, x->y, &zero, x->b, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, x->d_spmm)); if (preserve_b) CUD(cudaMemcpyAsync(x->d_b_check, x->d_b, sizeof(float) * static_cast<size_t>(a.cols) * l, cudaMemcpyDeviceToDevice, x->stream)); CUD(cudaEventRecord(x->b_end, x->stream));
  CUD(cudaEventRecord(x->svd_start, x->stream)); SOL(cusolverDnSgesvd(x->solver, 'N', 'N', a.cols, l, x->d_b, a.cols, x->d_sigma, nullptr, a.cols, nullptr, l, x->d_svd_work, x->svd_work, nullptr, x->d_info)); CUD(cudaEventRecord(x->svd_end, x->stream));
  return true;
}

static bool event_ms(cudaEvent_t start, cudaEvent_t end, double* out) { float x = 0.0F; if (!check_cuda(cudaEventElapsedTime(&x, start, end), __FILE__, __LINE__)) return false; *out = x; return true; }
static bool relative_b_error(const CsrMatrix& a, const std::vector<float>& h_q, const std::vector<float>& h_b, int l, double* out) {
  std::vector<float> ref(static_cast<size_t>(a.cols) * l, 0.0F);
  for (int r = 0; r < a.rows; ++r) for (int j = a.offsets[r]; j < a.offsets[r + 1]; ++j) for (int c = 0; c < l; ++c) {
    ref[a.columns[j] + static_cast<size_t>(c) * a.cols] += a.values[j] * h_q[r + static_cast<size_t>(c) * a.rows];
  }
  double d2 = 0.0, r2 = 0.0; for (size_t i = 0; i < ref.size(); ++i) { const double d = static_cast<double>(h_b[i]) - ref[i]; d2 += d * d; r2 += static_cast<double>(ref[i]) * ref[i]; }
  *out = std::sqrt(d2 / std::max(r2, 1e-30)); return true;
}

static double intersection(double a0, double a1, double b0, double b1) { return std::max(0.0, std::min(a1, b1) - std::max(a0, b0)); }

int main(int argc, char** argv) {
  const std::string input = argc >= 2 ? argv[1] : "synthetic"; const std::string output = argc >= 3 ? argv[2] : "stage59_end_to_end_overlap.csv"; const int rank = argc >= 4 ? std::atoi(argv[3]) : 16; const int l = rank + 8;
  CsrMatrix a; if (input == "synthetic") a = synthetic(2048, 1536, 8); else if (!read_mtx(input, &a)) { std::fprintf(stderr, "Cannot read Matrix Market input: %s\n", input.c_str()); return 65; }
  if (rank <= 0 || l > std::min(a.rows, a.cols)) return 65;
  if (!check_cuda(cudaSetDevice(0), __FILE__, __LINE__)) return 1;
  int *d_offsets = nullptr, *d_columns = nullptr; float* d_values = nullptr; if (!check_cuda(cudaMalloc(&d_offsets, sizeof(int) * a.offsets.size()), __FILE__, __LINE__) || !check_cuda(cudaMalloc(&d_columns, sizeof(int) * a.columns.size()), __FILE__, __LINE__) || !check_cuda(cudaMalloc(&d_values, sizeof(float) * a.values.size()), __FILE__, __LINE__)) return 1;
  if (!check_cuda(cudaMemcpy(d_offsets, a.offsets.data(), sizeof(int) * a.offsets.size(), cudaMemcpyHostToDevice), __FILE__, __LINE__) || !check_cuda(cudaMemcpy(d_columns, a.columns.data(), sizeof(int) * a.columns.size(), cudaMemcpyHostToDevice), __FILE__, __LINE__) || !check_cuda(cudaMemcpy(d_values, a.values.data(), sizeof(float) * a.values.size(), cudaMemcpyHostToDevice), __FILE__, __LINE__)) return 1;
  Lane lane[2]; if (!init_lane(&lane[0], a, l, d_offsets, d_columns, d_values) || !init_lane(&lane[1], a, l, d_offsets, d_columns, d_values)) { destroy_lane(&lane[0]); destroy_lane(&lane[1]); cudaFree(d_values); cudaFree(d_columns); cudaFree(d_offsets); return 1; }
  // Warm up CUDA, cuSPARSE, and cuSOLVER before timing either schedule.
  for (int i = 0; i < 2; ++i) { if (!enqueue_lane(&lane[i], a, l, 0x1000u + i, false) || !check_cuda(cudaStreamSynchronize(lane[i].stream), __FILE__, __LINE__)) return 1; }
  double sequential_ms = 0.0; auto seq_start = std::chrono::steady_clock::now(); for (int i = 0; i < 2; ++i) { if (!enqueue_lane(&lane[i], a, l, 0x20260928u + i, false) || !check_cuda(cudaStreamSynchronize(lane[i].stream), __FILE__, __LINE__)) return 1; } sequential_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - seq_start).count();
  cudaEvent_t overlap_anchor = nullptr; const auto overlap_start = std::chrono::steady_clock::now();
  if (!check_cuda(cudaEventCreate(&overlap_anchor), __FILE__, __LINE__) ||
      !check_cuda(cudaEventRecord(overlap_anchor, 0), __FILE__, __LINE__) ||
      !check_cuda(cudaStreamWaitEvent(lane[0].stream, overlap_anchor, 0), __FILE__, __LINE__) ||
      !check_cuda(cudaStreamWaitEvent(lane[1].stream, overlap_anchor, 0), __FILE__, __LINE__) ||
      !enqueue_lane(&lane[0], a, l, 0xA510u, true) ||
      !enqueue_lane(&lane[1], a, l, 0xB620u, false)) return 1;
  // Synchronize each lane separately so a concurrent cuSPARSE/cuSOLVER
  // failure identifies the responsible stream in the uploaded log.  A device
  // wide synchronize only reports a generic code and hid which lane failed.
  const cudaError_t lane0_status = cudaStreamSynchronize(lane[0].stream);
  const cudaError_t lane1_status = cudaStreamSynchronize(lane[1].stream);
  if (lane0_status != cudaSuccess || lane1_status != cudaSuccess) {
    check_cuda(lane0_status, __FILE__, __LINE__);
    check_cuda(lane1_status, __FILE__, __LINE__);
    std::fprintf(stderr, "Concurrent overlap failed; rerun with CUDA_LAUNCH_BLOCKING=1 for an operation-level trace.\n");
    return 1;
  }
  const double overlap_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - overlap_start).count();
  double q0 = 0.0, qr0 = 0.0, b0 = 0.0, svd0 = 0.0, q1 = 0.0, qr1 = 0.0, b1 = 0.0, svd1 = 0.0; event_ms(lane[0].q_start, lane[0].q_end, &q0); event_ms(lane[0].qr_start, lane[0].qr_end, &qr0); event_ms(lane[0].b_start, lane[0].b_end, &b0); event_ms(lane[0].svd_start, lane[0].svd_end, &svd0); event_ms(lane[1].q_start, lane[1].q_end, &q1); event_ms(lane[1].qr_start, lane[1].qr_end, &qr1); event_ms(lane[1].b_start, lane[1].b_end, &b1); event_ms(lane[1].svd_start, lane[1].svd_end, &svd1);
  double qr0_start = 0.0, qr1_start = 0.0, b0_start = 0.0, b1_start = 0.0, svd0_start = 0.0, svd1_start = 0.0, svd0_end = 0.0, svd1_end = 0.0;
  if (!event_ms(lane[0].run_start, lane[0].qr_start, &qr0_start) || !event_ms(lane[1].run_start, lane[1].qr_start, &qr1_start) || !event_ms(lane[0].run_start, lane[0].b_start, &b0_start) || !event_ms(lane[1].run_start, lane[1].b_start, &b1_start) || !event_ms(lane[0].run_start, lane[0].svd_start, &svd0_start) || !event_ms(lane[1].run_start, lane[1].svd_start, &svd1_start) || !event_ms(lane[0].run_start, lane[0].svd_end, &svd0_end) || !event_ms(lane[1].run_start, lane[1].svd_end, &svd1_end)) return 1;
  double rel_error = 0.0; std::vector<float> h_b(static_cast<size_t>(a.cols) * l); std::vector<float> h_q(static_cast<size_t>(a.rows) * l); if (!check_cuda(cudaMemcpy(h_b.data(), lane[0].d_b_check, sizeof(float) * h_b.size(), cudaMemcpyDeviceToHost), __FILE__, __LINE__) || !check_cuda(cudaMemcpy(h_q.data(), lane[0].d_y, sizeof(float) * h_q.size(), cudaMemcpyDeviceToHost), __FILE__, __LINE__)) return 1; relative_b_error(a, h_q, h_b, l, &rel_error);
  std::vector<float> h_qtq(static_cast<size_t>(l) * l); const float one = 1.0F, zero = 0.0F; BLAS(cublasSgemm(lane[0].blas, CUBLAS_OP_T, CUBLAS_OP_N, l, l, a.rows, &one, lane[0].d_y, a.rows, lane[0].d_y, a.rows, &zero, lane[0].d_qtq, l)); if (!check_cuda(cudaStreamSynchronize(lane[0].stream), __FILE__, __LINE__) || !check_cuda(cudaMemcpy(h_qtq.data(), lane[0].d_qtq, sizeof(float) * h_qtq.size(), cudaMemcpyDeviceToHost), __FILE__, __LINE__)) return 1; double orth = 0.0; for (int c = 0; c < l; ++c) for (int r = 0; r < l; ++r) orth = std::max(orth, static_cast<double>(std::fabs(h_qtq[r + static_cast<size_t>(c) * l] - (r == c ? 1.0F : 0.0F))));
  int info0 = 0, info1 = 0; check_cuda(cudaMemcpy(&info0, lane[0].d_info, sizeof(int), cudaMemcpyDeviceToHost), __FILE__, __LINE__); check_cuda(cudaMemcpy(&info1, lane[1].d_info, sizeof(int), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
  double overlap_qr_svd = intersection(qr0_start, svd0_end, qr1_start, svd1_end); double overlap_svd_b = intersection(svd0_start, svd0_end, svd1_start, svd1_end); const double speedup = sequential_ms / std::max(overlap_ms, 1e-9);
  std::ofstream out(output); out << "input,rows,cols,nnz,rank,lanes,sequential_ms,overlap_ms,speedup,q0_ms,qr0_ms,b0_ms,svd0_ms,q1_ms,qr1_ms,b1_ms,svd1_ms,qr_to_svd_envelope_overlap_ms,svd_stage_overlap_ms,completed_lanes,solver_info0,solver_info1,q_orthogonality_error,atq_relative_error\n"; out << input << ',' << a.rows << ',' << a.cols << ',' << a.values.size() << ',' << rank << ",2," << sequential_ms << ',' << overlap_ms << ',' << speedup << ',' << q0 << ',' << qr0 << ',' << b0 << ',' << svd0 << ',' << q1 << ',' << qr1 << ',' << b1 << ',' << svd1 << ',' << overlap_qr_svd << ',' << overlap_svd_b << ",2," << info0 << ',' << info1 << ',' << orth << ',' << rel_error << '\n';
  std::printf("End-to-end overlap: A(%d x %d) lanes=2 sequential=%.3f ms overlap=%.3f ms speedup=%.3fx QR/SVD-window=%.3f ms B-window=%.3f ms offsets(qr=%.3f/%.3f,svd=%.3f/%.3f,ends=%.3f/%.3f) Q_orth=%.3e ATQ_error=%.3e info=%d/%d\n", a.rows, a.cols, sequential_ms, overlap_ms, speedup, overlap_qr_svd, overlap_svd_b, qr0_start, qr1_start, svd0_start, svd1_start, svd0_end, svd1_end, orth, rel_error, info0, info1);
  destroy_lane(&lane[0]); destroy_lane(&lane[1]); cudaEventDestroy(overlap_anchor); cudaFree(d_values); cudaFree(d_columns); cudaFree(d_offsets); return (info0 != 0 || info1 != 0 || orth > 1e-3 || rel_error > 1e-4) ? 2 : 0;
}
