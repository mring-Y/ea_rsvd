#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverDn.h>
#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#define CUDA_CHECK(call) do { \
  const cudaError_t status = (call); \
  if (status != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
    return 1; \
  } \
} while (0)
#define CUBLAS_CHECK(call) do { \
  const cublasStatus_t status = (call); \
  if (status != CUBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, static_cast<int>(status)); \
    return 1; \
  } \
} while (0)
#define CUSOLVER_CHECK(call) do { \
  const cusolverStatus_t status = (call); \
  if (status != CUSOLVER_STATUS_SUCCESS) { \
    std::fprintf(stderr, "cuSOLVER error at %s:%d: %d\n", __FILE__, __LINE__, static_cast<int>(status)); \
    return 1; \
  } \
} while (0)
#define CUSPARSE_CHECK(call) do { \
  const cusparseStatus_t status = (call); \
  if (status != CUSPARSE_STATUS_SUCCESS) { \
    std::fprintf(stderr, "cuSPARSE error at %s:%d: %d\n", __FILE__, __LINE__, static_cast<int>(status)); \
    return 1; \
  } \
} while (0)
#define CURAND_CHECK(call) do { \
  const curandStatus_t status = (call); \
  if (status != CURAND_STATUS_SUCCESS) { \
    std::fprintf(stderr, "cuRAND error at %s:%d: %d\n", __FILE__, __LINE__, static_cast<int>(status)); \
    return 1; \
  } \
} while (0)

struct CsrMatrix {
  int rows = 0;
  int cols = 0;
  std::vector<int> offsets;
  std::vector<int> columns;
  std::vector<float> values;
};

struct EventPair {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;

  bool create() {
    return cudaEventCreate(&start) == cudaSuccess && cudaEventCreate(&stop) == cudaSuccess;
  }

  void destroy() {
    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
    start = nullptr;
    stop = nullptr;
  }
};

template <typename Operation>
static bool measure_event(EventPair& pair, Operation&& operation, double* elapsed_ms) {
  if (cudaEventRecord(pair.start) != cudaSuccess) return false;
  operation();
  if (cudaEventRecord(pair.stop) != cudaSuccess || cudaEventSynchronize(pair.stop) != cudaSuccess) return false;
  float measured = 0.0F;
  if (cudaEventElapsedTime(&measured, pair.start, pair.stop) != cudaSuccess) return false;
  *elapsed_ms += static_cast<double>(measured);
  return true;
}

static bool read_matrix_market(const std::string& path, CsrMatrix* matrix) {
  std::ifstream in(path);
  std::string line;
  if (!in || !std::getline(in, line)) return false;
  if (line.rfind("%%MatrixMarket matrix coordinate", 0) != 0 || line.find("general") == std::string::npos) {
    std::fprintf(stderr, "Only general Matrix Market coordinate matrices are supported: %s\n", path.c_str());
    return false;
  }
  do {
    if (!std::getline(in, line)) return false;
  } while (line.empty() || line[0] == '%');
  std::istringstream shape(line);
  int nnz = 0;
  if (!(shape >> matrix->rows >> matrix->cols >> nnz) || matrix->rows <= 0 || matrix->cols <= 0 || nnz < 0) {
    return false;
  }
  std::vector<std::vector<std::pair<int, float>>> rows(matrix->rows);
  for (int i = 0; i < nnz; ++i) {
    if (!std::getline(in, line)) return false;
    std::istringstream entry(line);
    int row = 0, col = 0;
    float value = 1.0F;
    if (!(entry >> row >> col) || row < 1 || row > matrix->rows || col < 1 || col > matrix->cols) return false;
    entry >> value;  // Pattern entries retain the default value.
    rows[row - 1].emplace_back(col - 1, value);
  }
  matrix->offsets.assign(static_cast<size_t>(matrix->rows) + 1U, 0);
  matrix->columns.reserve(nnz);
  matrix->values.reserve(nnz);
  for (int row = 0; row < matrix->rows; ++row) {
    std::sort(rows[row].begin(), rows[row].end());
    matrix->offsets[row + 1] = matrix->offsets[row] + static_cast<int>(rows[row].size());
    for (const auto& entry : rows[row]) {
      matrix->columns.push_back(entry.first);
      matrix->values.push_back(entry.second);
    }
  }
  return true;
}

static bool run_spmm(cusparseHandle_t handle, cusparseOperation_t op_a, cusparseSpMatDescr_t a,
                     cusparseDnMatDescr_t x, cusparseDnMatDescr_t y, float alpha, float beta,
                     void* buffer) {
  return cusparseSpMM(handle, op_a, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, a, x, &beta, y,
                       CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buffer) == CUSPARSE_STATUS_SUCCESS;
}

int main(int argc, char** argv) {
  if (argc < 2 || argc > 7) {
    std::fprintf(stderr, "Usage: gpu_sparse_reference <matrix.mtx> [results.csv] [rank] [oversampling] [power_steps:0..2] [seed]\n");
    return 64;
  }
  const std::string input = argv[1];
  const std::string csv = argc >= 3 ? argv[2] : "results/logs/gpu_sparse_reference.csv";
  const int rank = argc >= 4 ? std::atoi(argv[3]) : 16;
  const int oversampling = argc >= 5 ? std::atoi(argv[4]) : 8;
  const int power_steps = argc >= 6 ? std::atoi(argv[5]) : 1;
  const unsigned long long seed = argc >= 7 ? std::strtoull(argv[6], nullptr, 10) : 20261003ULL;

  CsrMatrix h;
  if (!read_matrix_market(input, &h)) return 65;
  const int width = rank + oversampling;
  if (rank <= 0 || oversampling < 0 || power_steps < 0 || power_steps > 2 || width > std::min(h.rows, h.cols)) {
    std::fprintf(stderr, "Invalid rank/oversampling/power depth for %d x %d matrix.\n", h.rows, h.cols);
    return 66;
  }

  const size_t nnz = h.values.size();
  const size_t omega_count = static_cast<size_t>(h.cols) * static_cast<size_t>(width);
  int* d_offsets = nullptr;
  int* d_columns = nullptr;
  int* d_info = nullptr;
  float* d_values = nullptr;
  float* d_omega = nullptr;
  float* d_q = nullptr;
  float* d_z = nullptr;
  float* d_b = nullptr;
  float* d_sigma = nullptr;
  float* d_tau_q = nullptr;
  float* d_tau_z = nullptr;
  float* d_qtq = nullptr;
  float* d_work = nullptr;
  float* d_svd_work = nullptr;
  void* d_spmm_buffer = nullptr;
  cusparseHandle_t sparse = nullptr;
  cusolverDnHandle_t solver = nullptr;
  cublasHandle_t blas = nullptr;
  curandGenerator_t rng = nullptr;
  cusparseSpMatDescr_t a = nullptr;
  cusparseDnMatDescr_t omega = nullptr, q = nullptr, z = nullptr, b = nullptr;
  EventPair h2d_events, spmm_forward_events, spmm_transpose_events, qr_events, svd_events, total_events;

  CUDA_CHECK(cudaMalloc(&d_offsets, sizeof(int) * h.offsets.size()));
  CUDA_CHECK(cudaMalloc(&d_columns, sizeof(int) * nnz));
  CUDA_CHECK(cudaMalloc(&d_values, sizeof(float) * nnz));
  CUDA_CHECK(cudaMalloc(&d_omega, sizeof(float) * (omega_count + (omega_count & 1U))));
  CUDA_CHECK(cudaMalloc(&d_q, sizeof(float) * static_cast<size_t>(h.rows) * width));
  CUDA_CHECK(cudaMalloc(&d_z, sizeof(float) * static_cast<size_t>(h.cols) * width));
  CUDA_CHECK(cudaMalloc(&d_b, sizeof(float) * static_cast<size_t>(h.cols) * width));
  CUDA_CHECK(cudaMalloc(&d_sigma, sizeof(float) * width));
  CUDA_CHECK(cudaMalloc(&d_tau_q, sizeof(float) * width));
  CUDA_CHECK(cudaMalloc(&d_tau_z, sizeof(float) * width));
  CUDA_CHECK(cudaMalloc(&d_qtq, sizeof(float) * static_cast<size_t>(width) * width));
  CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
  CUSPARSE_CHECK(cusparseCreate(&sparse));
  CUSOLVER_CHECK(cusolverDnCreate(&solver));
  CUBLAS_CHECK(cublasCreate(&blas));
  CURAND_CHECK(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rng, seed));
  CURAND_CHECK(curandGenerateNormal(rng, d_omega, static_cast<size_t>(omega_count + (omega_count & 1U)), 0.0F, 1.0F));

  CUSPARSE_CHECK(cusparseCreateCsr(&a, h.rows, h.cols, static_cast<int64_t>(nnz), d_offsets, d_columns, d_values,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  CUSPARSE_CHECK(cusparseCreateDnMat(&omega, h.cols, width, h.cols, d_omega, CUDA_R_32F, CUSPARSE_ORDER_COL));
  CUSPARSE_CHECK(cusparseCreateDnMat(&q, h.rows, width, h.rows, d_q, CUDA_R_32F, CUSPARSE_ORDER_COL));
  CUSPARSE_CHECK(cusparseCreateDnMat(&z, h.cols, width, h.cols, d_z, CUDA_R_32F, CUSPARSE_ORDER_COL));
  CUSPARSE_CHECK(cusparseCreateDnMat(&b, h.cols, width, h.cols, d_b, CUDA_R_32F, CUSPARSE_ORDER_COL));

  const float one = 1.0F, zero = 0.0F;
  size_t forward_bytes = 0, transpose_bytes = 0;
  CUSPARSE_CHECK(cusparseSpMM_bufferSize(sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &one, a, omega, &zero, q, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &forward_bytes));
  CUSPARSE_CHECK(cusparseSpMM_bufferSize(sparse, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &one, a, q, &zero, z, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &transpose_bytes));
  CUDA_CHECK(cudaMalloc(&d_spmm_buffer, std::max(forward_bytes, transpose_bytes)));

  int q_geqrf_work = 0, q_orgqr_work = 0, z_geqrf_work = 0, z_orgqr_work = 0, svd_work = 0;
  CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(solver, h.rows, width, d_q, h.rows, &q_geqrf_work));
  CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(solver, h.rows, width, width, d_q, h.rows, d_tau_q, &q_orgqr_work));
  CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(solver, h.cols, width, d_z, h.cols, &z_geqrf_work));
  CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(solver, h.cols, width, width, d_z, h.cols, d_tau_z, &z_orgqr_work));
  CUSOLVER_CHECK(cusolverDnSgesvd_bufferSize(solver, h.cols, width, &svd_work));
  CUDA_CHECK(cudaMalloc(&d_work, sizeof(float) * static_cast<size_t>(std::max({q_geqrf_work, q_orgqr_work, z_geqrf_work, z_orgqr_work}))));
  CUDA_CHECK(cudaMalloc(&d_svd_work, sizeof(float) * static_cast<size_t>(svd_work)));
  if (!h2d_events.create() || !spmm_forward_events.create() || !spmm_transpose_events.create() ||
      !qr_events.create() || !svd_events.create() || !total_events.create()) {
    std::fprintf(stderr, "CUDA event creation failed.\n");
    return 1;
  }

  double h2d_ms = 0.0, spmm_forward_ms = 0.0, spmm_transpose_ms = 0.0, qr_ms = 0.0, svd_ms = 0.0;
  bool operation_ok = true;
  if (!measure_event(h2d_events, [&]() {
        operation_ok = cudaMemcpy(d_offsets, h.offsets.data(), sizeof(int) * h.offsets.size(), cudaMemcpyHostToDevice) == cudaSuccess;
        operation_ok = operation_ok && cudaMemcpy(d_columns, h.columns.data(), sizeof(int) * nnz, cudaMemcpyHostToDevice) == cudaSuccess;
        operation_ok = operation_ok && cudaMemcpy(d_values, h.values.data(), sizeof(float) * nnz, cudaMemcpyHostToDevice) == cudaSuccess;
      }, &h2d_ms) || !operation_ok) {
    std::fprintf(stderr, "CSR host-to-device copy failed.\n");
    return 1;
  }

  if (cudaEventRecord(total_events.start) != cudaSuccess) return 1;
  if (!measure_event(spmm_forward_events, [&]() {
        operation_ok = run_spmm(sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, a, omega, q, one, zero, d_spmm_buffer);
      }, &spmm_forward_ms) || !operation_ok) return 1;
  if (!measure_event(qr_events, [&]() {
        operation_ok = cusolverDnSgeqrf(solver, h.rows, width, d_q, h.rows, d_tau_q, d_work, q_geqrf_work, d_info) == CUSOLVER_STATUS_SUCCESS;
        operation_ok = operation_ok && cusolverDnSorgqr(solver, h.rows, width, width, d_q, h.rows, d_tau_q, d_work, q_orgqr_work, d_info) == CUSOLVER_STATUS_SUCCESS;
      }, &qr_ms) || !operation_ok) return 1;

  for (int step = 0; step < power_steps; ++step) {
    if (!measure_event(spmm_transpose_events, [&]() {
          operation_ok = run_spmm(sparse, CUSPARSE_OPERATION_TRANSPOSE, a, q, z, one, zero, d_spmm_buffer);
        }, &spmm_transpose_ms) || !operation_ok) return 1;
    if (!measure_event(qr_events, [&]() {
          operation_ok = cusolverDnSgeqrf(solver, h.cols, width, d_z, h.cols, d_tau_z, d_work, z_geqrf_work, d_info) == CUSOLVER_STATUS_SUCCESS;
          operation_ok = operation_ok && cusolverDnSorgqr(solver, h.cols, width, width, d_z, h.cols, d_tau_z, d_work, z_orgqr_work, d_info) == CUSOLVER_STATUS_SUCCESS;
        }, &qr_ms) || !operation_ok) return 1;
    if (!measure_event(spmm_forward_events, [&]() {
          operation_ok = run_spmm(sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, a, z, q, one, zero, d_spmm_buffer);
        }, &spmm_forward_ms) || !operation_ok) return 1;
    if (!measure_event(qr_events, [&]() {
          operation_ok = cusolverDnSgeqrf(solver, h.rows, width, d_q, h.rows, d_tau_q, d_work, q_geqrf_work, d_info) == CUSOLVER_STATUS_SUCCESS;
          operation_ok = operation_ok && cusolverDnSorgqr(solver, h.rows, width, width, d_q, h.rows, d_tau_q, d_work, q_orgqr_work, d_info) == CUSOLVER_STATUS_SUCCESS;
        }, &qr_ms) || !operation_ok) return 1;
  }
  if (!measure_event(spmm_transpose_events, [&]() {
        operation_ok = run_spmm(sparse, CUSPARSE_OPERATION_TRANSPOSE, a, q, b, one, zero, d_spmm_buffer);
      }, &spmm_transpose_ms) || !operation_ok) return 1;
  if (!measure_event(svd_events, [&]() {
        operation_ok = cusolverDnSgesvd(solver, 'N', 'N', h.cols, width, d_b, h.cols, d_sigma,
                                         nullptr, h.cols, nullptr, width, d_svd_work, svd_work, nullptr, d_info) == CUSOLVER_STATUS_SUCCESS;
      }, &svd_ms) || !operation_ok) return 1;
  CUBLAS_CHECK(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, width, width, h.rows, &one,
                           d_q, h.rows, d_q, h.rows, &zero, d_qtq, width));
  if (cudaEventRecord(total_events.stop) != cudaSuccess || cudaEventSynchronize(total_events.stop) != cudaSuccess) return 1;
  float total_ms_float = 0.0F;
  if (cudaEventElapsedTime(&total_ms_float, total_events.start, total_events.stop) != cudaSuccess) return 1;

  int solver_info = 0;
  CUDA_CHECK(cudaMemcpy(&solver_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (solver_info != 0) {
    std::fprintf(stderr, "cuSOLVER returned info=%d.\n", solver_info);
    return 2;
  }
  std::vector<float> h_sigma(width), h_qtq(static_cast<size_t>(width) * width);
  CUDA_CHECK(cudaMemcpy(h_sigma.data(), d_sigma, sizeof(float) * width, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_qtq.data(), d_qtq, sizeof(float) * h_qtq.size(), cudaMemcpyDeviceToHost));
  double matrix_norm_sq = 0.0;
  for (float value : h.values) matrix_norm_sq += static_cast<double>(value) * static_cast<double>(value);
  double captured_rank_sq = 0.0;
  for (int i = 0; i < rank; ++i) captured_rank_sq += static_cast<double>(h_sigma[i]) * static_cast<double>(h_sigma[i]);
  const double rank_residual = std::sqrt(std::max(0.0, matrix_norm_sq - captured_rank_sq) / std::max(matrix_norm_sq, 1e-300));
  double q_orth_error = 0.0;
  for (int c = 0; c < width; ++c) for (int r = 0; r < width; ++r) {
    const double expected = r == c ? 1.0 : 0.0;
    q_orth_error = std::max(q_orth_error, std::fabs(static_cast<double>(h_qtq[r + c * width]) - expected));
  }

  const std::filesystem::path csv_path(csv);
  if (!csv_path.parent_path().empty()) std::filesystem::create_directories(csv_path.parent_path());
  const bool new_csv = !std::filesystem::exists(csv_path);
  std::ofstream out(csv, std::ios::app);
  if (!out) {
    std::fprintf(stderr, "Could not open output CSV: %s\n", csv.c_str());
    return 1;
  }
  if (new_csv) {
    out << "baseline,input,rows,cols,nnz,rank,oversampling,width,power_steps,seed,h2d_ms,spmm_forward_ms,spmm_transpose_ms,qr_ms,svd_ms,gpu_pipeline_ms,rank_k_residual,q_orthogonality_max_error,matrix_frobenius_sq\n";
  }
  out << "cusparse_cusolver," << input << ',' << h.rows << ',' << h.cols << ',' << nnz << ',' << rank << ','
      << oversampling << ',' << width << ',' << power_steps << ',' << seed << ',' << h2d_ms << ',' << spmm_forward_ms << ','
      << spmm_transpose_ms << ',' << qr_ms << ',' << svd_ms << ',' << total_ms_float << ',' << rank_residual << ','
      << q_orth_error << ',' << matrix_norm_sq << '\n';
  std::printf("same-GPU sparse reference: %s, A(%d x %d), nnz=%zu, rank=%d, width=%d, q=%d\n",
              input.c_str(), h.rows, h.cols, nnz, rank, width, power_steps);
  std::printf("GPU events (ms): forward SpMM %.3f, transpose SpMM %.3f, QR %.3f, SVD %.3f, total %.3f\n",
              spmm_forward_ms, spmm_transpose_ms, qr_ms, svd_ms, static_cast<double>(total_ms_float));
  std::printf("rank-%d residual %.8f, Q orthogonality %.3e\n", rank, rank_residual, q_orth_error);

  h2d_events.destroy(); spmm_forward_events.destroy(); spmm_transpose_events.destroy(); qr_events.destroy(); svd_events.destroy(); total_events.destroy();
  cudaFree(d_svd_work); cudaFree(d_work); cudaFree(d_spmm_buffer); cudaFree(d_qtq); cudaFree(d_info); cudaFree(d_tau_z); cudaFree(d_tau_q);
  cudaFree(d_sigma); cudaFree(d_b); cudaFree(d_z); cudaFree(d_q); cudaFree(d_omega); cudaFree(d_values); cudaFree(d_columns); cudaFree(d_offsets);
  cusparseDestroyDnMat(b); cusparseDestroyDnMat(z); cusparseDestroyDnMat(q); cusparseDestroyDnMat(omega); cusparseDestroySpMat(a);
  curandDestroyGenerator(rng); cublasDestroy(blas); cusolverDnDestroy(solver); cusparseDestroy(sparse);
  return 0;
}
