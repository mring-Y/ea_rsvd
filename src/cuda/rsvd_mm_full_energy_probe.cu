#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverDn.h>
#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { const cudaError_t s = (call); if (s != cudaSuccess) { std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(s)); return 1; } } while (0)
#define CUSPARSE_CHECK(call) do { const cusparseStatus_t s = (call); if (s != CUSPARSE_STATUS_SUCCESS) { std::fprintf(stderr, "cuSPARSE error: %d\n", static_cast<int>(s)); return 1; } } while (0)
#define CUSOLVER_CHECK(call) do { const cusolverStatus_t s = (call); if (s != CUSOLVER_STATUS_SUCCESS) { std::fprintf(stderr, "cuSOLVER error: %d\n", static_cast<int>(s)); return 1; } } while (0)
#define CUBLAS_CHECK(call) do { const cublasStatus_t s = (call); if (s != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "cuBLAS error: %d\n", static_cast<int>(s)); return 1; } } while (0)
#define CURAND_CHECK(call) do { const curandStatus_t s = (call); if (s != CURAND_STATUS_SUCCESS) { std::fprintf(stderr, "cuRAND error: %d\n", static_cast<int>(s)); return 1; } } while (0)

struct CsrMatrix {
  int rows = 0, cols = 0;
  std::vector<int> offsets, columns;
  std::vector<float> values;
};

bool read_matrix_market(const std::string& path, CsrMatrix* matrix) {
  std::ifstream in(path);
  std::string line;
  if (!in || !std::getline(in, line) || line.rfind("%%MatrixMarket matrix coordinate", 0) != 0 ||
      line.find("general") == std::string::npos) {
    std::fprintf(stderr, "Only general Matrix Market coordinate matrices are supported: %s\n", path.c_str());
    return false;
  }
  do { if (!std::getline(in, line)) return false; } while (line.empty() || line[0] == '%');
  std::istringstream shape(line);
  int nnz = 0;
  if (!(shape >> matrix->rows >> matrix->cols >> nnz) || matrix->rows <= 0 || matrix->cols <= 0 || nnz < 0) return false;
  std::vector<std::vector<std::pair<int, float>>> rows(matrix->rows);
  for (int i = 0; i < nnz; ++i) {
    if (!std::getline(in, line)) return false;
    std::istringstream entry(line);
    int row = 0, col = 0;
    float value = 1.0F;
    if (!(entry >> row >> col) || row < 1 || row > matrix->rows || col < 1 || col > matrix->cols) return false;
    entry >> value;  // Pattern matrices retain the default value.
    rows[row - 1].emplace_back(col - 1, value);
  }
  matrix->offsets.assign(matrix->rows + 1, 0);
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

int main(int argc, char** argv) {
  if (argc < 2 || argc > 7) {
    std::fprintf(stderr, "Usage: rsvd_mm_benchmark <matrix.mtx> [results.csv] [rank] [oversampling] [power_steps:0..2] [seed]\n");
    return 64;
  }
  const std::string input = argv[1];
  const std::string csv = argc >= 3 ? argv[2] : "results/logs/rsvd_stage2.csv";
  const int target_rank = argc >= 4 ? std::atoi(argv[3]) : 2;
  const int oversampling = argc >= 5 ? std::atoi(argv[4]) : 2;
  const int power_steps = argc >= 6 ? std::atoi(argv[5]) : 1;
  const unsigned long long seed = argc >= 7 ? std::strtoull(argv[6], nullptr, 10) : 20260730ULL;
  CsrMatrix h;
  const auto read_start = std::chrono::steady_clock::now();
  if (!read_matrix_market(input, &h)) return 65;
  const auto read_end = std::chrono::steady_clock::now();
  const int l = target_rank + oversampling;
  if (target_rank <= 0 || oversampling < 0 || power_steps < 0 || power_steps > 2 || l > std::min(h.rows, h.cols)) {
    std::fprintf(stderr, "Invalid rank/oversampling for %d x %d matrix.\n", h.rows, h.cols);
    return 66;
  }

  int *d_offsets = nullptr, *d_columns = nullptr, *d_info = nullptr;
  float *d_values = nullptr, *d_omega = nullptr, *d_q = nullptr, *d_tau = nullptr;
  float *d_projected = nullptr, *d_sigma = nullptr, *d_sigma_previous = nullptr, *d_qtq = nullptr, *d_work = nullptr, *d_svd_work = nullptr;
  constexpr int probe_count = 16;
  float *d_probe = nullptr, *d_residual = nullptr, *d_coeff = nullptr, *d_z = nullptr, *d_tau_z = nullptr;
  void* d_spmm = nullptr;
  cusparseHandle_t sparse = nullptr; cusolverDnHandle_t solver = nullptr; cublasHandle_t blas = nullptr; curandGenerator_t rng = nullptr;
  cusparseSpMatDescr_t a = nullptr; cusparseDnMatDescr_t omega = nullptr, q = nullptr, projected = nullptr, z = nullptr, probe = nullptr, residual = nullptr;

  CUDA_CHECK(cudaMalloc(&d_offsets, sizeof(int) * h.offsets.size()));
  CUDA_CHECK(cudaMalloc(&d_columns, sizeof(int) * h.columns.size()));
  CUDA_CHECK(cudaMalloc(&d_values, sizeof(float) * h.values.size()));
  CUDA_CHECK(cudaMalloc(&d_omega, sizeof(float) * h.cols * l));
  CUDA_CHECK(cudaMalloc(&d_q, sizeof(float) * h.rows * l));
  CUDA_CHECK(cudaMalloc(&d_tau, sizeof(float) * l));
  CUDA_CHECK(cudaMalloc(&d_projected, sizeof(float) * h.cols * l));
  CUDA_CHECK(cudaMalloc(&d_sigma, sizeof(float) * l));
  CUDA_CHECK(cudaMalloc(&d_sigma_previous, sizeof(float) * l));
  CUDA_CHECK(cudaMalloc(&d_qtq, sizeof(float) * l * l));
  CUDA_CHECK(cudaMalloc(&d_probe, sizeof(float) * h.cols * probe_count));
  CUDA_CHECK(cudaMalloc(&d_residual, sizeof(float) * h.rows * probe_count));
  CUDA_CHECK(cudaMalloc(&d_coeff, sizeof(float) * l * probe_count));
  CUDA_CHECK(cudaMalloc(&d_z, sizeof(float) * h.cols * l));
  CUDA_CHECK(cudaMalloc(&d_tau_z, sizeof(float) * l));
  CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_offsets, h.offsets.data(), sizeof(int) * h.offsets.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_columns, h.columns.data(), sizeof(int) * h.columns.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h.values.data(), sizeof(float) * h.values.size(), cudaMemcpyHostToDevice));
  CUSPARSE_CHECK(cusparseCreate(&sparse)); CUSOLVER_CHECK(cusolverDnCreate(&solver)); CUBLAS_CHECK(cublasCreate(&blas));
  CURAND_CHECK(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rng, seed));
  CURAND_CHECK(curandGenerateNormal(rng, d_omega, h.cols * l, 0.0F, 1.0F));
  CURAND_CHECK(curandGenerateNormal(rng, d_probe, h.cols * probe_count, 0.0F, 1.0F));
  CUSPARSE_CHECK(cusparseCreateCsr(&a, h.rows, h.cols, static_cast<int64_t>(h.values.size()), d_offsets, d_columns, d_values,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  CUSPARSE_CHECK(cusparseCreateDnMat(&omega, h.cols, l, h.cols, d_omega, CUDA_R_32F, CUSPARSE_ORDER_COL));
  CUSPARSE_CHECK(cusparseCreateDnMat(&q, h.rows, l, h.rows, d_q, CUDA_R_32F, CUSPARSE_ORDER_COL));
  CUSPARSE_CHECK(cusparseCreateDnMat(&projected, h.cols, l, h.cols, d_projected, CUDA_R_32F, CUSPARSE_ORDER_COL));
  CUSPARSE_CHECK(cusparseCreateDnMat(&z, h.cols, l, h.cols, d_z, CUDA_R_32F, CUSPARSE_ORDER_COL));
  CUSPARSE_CHECK(cusparseCreateDnMat(&probe, h.cols, probe_count, h.cols, d_probe, CUDA_R_32F, CUSPARSE_ORDER_COL));
  CUSPARSE_CHECK(cusparseCreateDnMat(&residual, h.rows, probe_count, h.rows, d_residual, CUDA_R_32F, CUSPARSE_ORDER_COL));
  const float one = 1.0F, zero = 0.0F;
  size_t forward_bytes = 0, transpose_bytes = 0, probe_bytes = 0;
  CUSPARSE_CHECK(cusparseSpMM_bufferSize(sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &one, a, omega, &zero, q, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &forward_bytes));
  CUSPARSE_CHECK(cusparseSpMM_bufferSize(sparse, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &one, a, q, &zero, projected, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &transpose_bytes));
  CUSPARSE_CHECK(cusparseSpMM_bufferSize(sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &one, a, probe, &zero, residual, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &probe_bytes));
  CUDA_CHECK(cudaMalloc(&d_spmm, std::max(forward_bytes, std::max(transpose_bytes, probe_bytes))));
  int geqrf_work = 0, orgqr_work = 0, z_geqrf_work = 0, z_orgqr_work = 0, svd_work = 0;
  CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(solver, h.rows, l, d_q, h.rows, &geqrf_work));
  CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(solver, h.rows, l, l, d_q, h.rows, d_tau, &orgqr_work));
  CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(solver, h.cols, l, d_z, h.cols, &z_geqrf_work));
  CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(solver, h.cols, l, l, d_z, h.cols, d_tau_z, &z_orgqr_work));
  CUDA_CHECK(cudaMalloc(&d_work, sizeof(float) * std::max(std::max(geqrf_work, orgqr_work), std::max(z_geqrf_work, z_orgqr_work))));
  CUSOLVER_CHECK(cusolverDnSgesvd_bufferSize(solver, h.cols, l, &svd_work));
  CUDA_CHECK(cudaMalloc(&d_svd_work, sizeof(float) * svd_work));

  cudaEvent_t start = nullptr, stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop)); CUDA_CHECK(cudaEventRecord(start));
  CUSPARSE_CHECK(cusparseSpMM(sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &one, a, omega, &zero, q, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_spmm));
  CUSOLVER_CHECK(cusolverDnSgeqrf(solver, h.rows, l, d_q, h.rows, d_tau, d_work, geqrf_work, d_info));
  CUSOLVER_CHECK(cusolverDnSorgqr(solver, h.rows, l, l, d_q, h.rows, d_tau, d_work, orgqr_work, d_info));
  if (power_steps >= 1) {
  CUSPARSE_CHECK(cusparseSpMM(sparse, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &one, a, q, &zero, projected, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_spmm));
  CUSOLVER_CHECK(cusolverDnSgesvd(solver, 'N', 'N', h.cols, l, d_projected, h.cols, d_sigma_previous,
      nullptr, h.cols, nullptr, l, d_svd_work, svd_work, nullptr, d_info));
  CUSPARSE_CHECK(cusparseSpMM(sparse, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &one, a, q, &zero, z, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_spmm));
  CUSOLVER_CHECK(cusolverDnSgeqrf(solver, h.cols, l, d_z, h.cols, d_tau_z, d_work, z_geqrf_work, d_info));
  CUSOLVER_CHECK(cusolverDnSorgqr(solver, h.cols, l, l, d_z, h.cols, d_tau_z, d_work, z_orgqr_work, d_info));
  CUSPARSE_CHECK(cusparseSpMM(sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &one, a, z, &zero, q, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_spmm));
  CUSOLVER_CHECK(cusolverDnSgeqrf(solver, h.rows, l, d_q, h.rows, d_tau, d_work, geqrf_work, d_info));
  CUSOLVER_CHECK(cusolverDnSorgqr(solver, h.rows, l, l, d_q, h.rows, d_tau, d_work, orgqr_work, d_info));
  }
  CUSPARSE_CHECK(cusparseSpMM(sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &one, a, probe, &zero, residual, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_spmm));
  const float minus_one = -1.0F;
  CUBLAS_CHECK(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, l, probe_count, h.rows, &one,
      d_q, h.rows, d_residual, h.rows, &zero, d_coeff, l));
  CUBLAS_CHECK(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, h.rows, probe_count, l, &minus_one,
      d_q, h.rows, d_coeff, l, &one, d_residual, h.rows));
  CUSPARSE_CHECK(cusparseSpMM(sparse, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &one, a, q, &zero, projected, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_spmm));
  CUSOLVER_CHECK(cusolverDnSgesvd(solver, 'N', 'N', h.cols, l, d_projected, h.cols, d_sigma,
      nullptr, h.cols, nullptr, l, d_svd_work, svd_work, nullptr, d_info));
  if (power_steps == 2) {
    CUDA_CHECK(cudaMemcpy(d_sigma_previous, d_sigma, sizeof(float) * l, cudaMemcpyDeviceToDevice));
    CUSPARSE_CHECK(cusparseSpMM(sparse, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &one, a, q, &zero, z, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_spmm));
    CUSOLVER_CHECK(cusolverDnSgeqrf(solver, h.cols, l, d_z, h.cols, d_tau_z, d_work, z_geqrf_work, d_info));
    CUSOLVER_CHECK(cusolverDnSorgqr(solver, h.cols, l, l, d_z, h.cols, d_tau_z, d_work, z_orgqr_work, d_info));
    CUSPARSE_CHECK(cusparseSpMM(sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &one, a, z, &zero, q, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_spmm));
    CUSOLVER_CHECK(cusolverDnSgeqrf(solver, h.rows, l, d_q, h.rows, d_tau, d_work, geqrf_work, d_info));
    CUSOLVER_CHECK(cusolverDnSorgqr(solver, h.rows, l, l, d_q, h.rows, d_tau, d_work, orgqr_work, d_info));
    CUSPARSE_CHECK(cusparseSpMM(sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &one, a, probe, &zero, residual, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_spmm));
    CUBLAS_CHECK(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, l, probe_count, h.rows, &one,
        d_q, h.rows, d_residual, h.rows, &zero, d_coeff, l));
    CUBLAS_CHECK(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, h.rows, probe_count, l, &minus_one,
        d_q, h.rows, d_coeff, l, &one, d_residual, h.rows));
    CUSPARSE_CHECK(cusparseSpMM(sparse, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &one, a, q, &zero, projected, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_spmm));
    CUSOLVER_CHECK(cusolverDnSgesvd(solver, 'N', 'N', h.cols, l, d_projected, h.cols, d_sigma,
        nullptr, h.cols, nullptr, l, d_svd_work, svd_work, nullptr, d_info));
  }
  CUBLAS_CHECK(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, l, l, h.rows, &one, d_q, h.rows, d_q, h.rows, &zero, d_qtq, l));
  CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
  float gpu_ms = 0.0F; CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

  std::vector<float> qtq(l * l), sigma(l), sigma_previous(l), residual_samples(h.rows * probe_count);
  CUDA_CHECK(cudaMemcpy(qtq.data(), d_qtq, sizeof(float) * qtq.size(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(sigma.data(), d_sigma, sizeof(float) * sigma.size(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(sigma_previous.data(), d_sigma_previous, sizeof(float) * sigma_previous.size(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(residual_samples.data(), d_residual, sizeof(float) * residual_samples.size(), cudaMemcpyDeviceToHost));
  float orth_error = 0.0F;
  double matrix_energy = 0.0, spectrum_energy = 0.0, rank_energy = 0.0;
  for (int c = 0; c < l; ++c) for (int r = 0; r < l; ++r) orth_error = std::fmax(orth_error, std::fabs(qtq[r + c * l] - (r == c ? 1.0F : 0.0F)));
  for (float v : h.values) matrix_energy += static_cast<double>(v) * v;
  for (float v : sigma) spectrum_energy += static_cast<double>(v) * v;
  for (int i = 0; i < target_rank; ++i) rank_energy += static_cast<double>(sigma[i]) * sigma[i];
  const double energy_error = l == std::min(h.rows, h.cols) ? std::fabs(spectrum_energy - matrix_energy) / matrix_energy : -1.0;
  // B = Q^T A has singular values in sigma, so ||B||_F^2 is ||QQ^T A||_F^2.
  const double exact_projection_residual = std::sqrt(std::fmax(matrix_energy - spectrum_energy, 0.0) / matrix_energy);
  // The first target_rank singular values define the requested rank-k approximation.
  const double exact_rank_residual = std::sqrt(std::fmax(matrix_energy - rank_energy, 0.0) / matrix_energy);
  float residual_sample_energy = 0.0F;
  for (float value : residual_samples) residual_sample_energy += value * value;
  const float projection_residual_estimate = std::sqrt(residual_sample_energy / (probe_count * matrix_energy));
  float pve = power_steps == 0 ? std::numeric_limits<float>::infinity() : 0.0F;
  if (power_steps >= 1) for (int i = 0; i < target_rank; ++i) pve = std::fmax(pve, std::fabs(sigma[i] - sigma_previous[i]) / std::fmax(sigma[target_rank - 1], 1e-7F));
  const auto read_ms = std::chrono::duration<double, std::milli>(read_end - read_start).count();
  std::filesystem::create_directories(std::filesystem::path(csv).parent_path());
  const bool new_log = !std::filesystem::exists(csv);
  std::ofstream out(csv, std::ios::app);
  if (new_log) out << "input,rows,cols,nnz,target_rank,oversampling,power_steps,seed,cpu_read_ms,gpu_pipeline_ms,q_orthogonality_max_error,full_sketch_energy_relative_error,projection_residual_estimate,exact_projection_residual,exact_rank_residual,probe_count,pve_one_power_step\n";
  out << input << ',' << h.rows << ',' << h.cols << ',' << h.values.size() << ',' << target_rank << ',' << oversampling << ','
      << power_steps << ',' << seed << ',' << read_ms << ',' << gpu_ms << ',' << orth_error << ',' << energy_error << ',' << projection_residual_estimate << ',' << exact_projection_residual << ',' << exact_rank_residual << ',' << probe_count << ',' << pve << '\n';
  std::printf("Matrix Market rSVD: %s, A(%d x %d), nnz=%zu, rank=%d, oversampling=%d\n", input.c_str(), h.rows, h.cols, h.values.size(), target_rank, oversampling);
  std::printf("GPU pipeline: %.3f ms; Q orthogonality: %.3e; probe rho_l: %.3e; exact rho_l: %.3e; exact rho_k: %.3e; PVE: %.3e\n", gpu_ms, orth_error, projection_residual_estimate, exact_projection_residual, exact_rank_residual, pve);
  if (orth_error > 1e-4F || (energy_error >= 0.0F && energy_error > 1e-4F)) return 2;
  std::puts("Matrix Market rSVD validation: PASS");

  cudaEventDestroy(stop); cudaEventDestroy(start); cudaFree(d_svd_work); cudaFree(d_work); cudaFree(d_spmm); cudaFree(d_info);
  cudaFree(d_tau_z); cudaFree(d_z); cudaFree(d_coeff); cudaFree(d_residual); cudaFree(d_probe); cudaFree(d_qtq); cudaFree(d_sigma_previous); cudaFree(d_sigma); cudaFree(d_projected); cudaFree(d_tau); cudaFree(d_q); cudaFree(d_omega);
  cudaFree(d_values); cudaFree(d_columns); cudaFree(d_offsets);
  cusparseDestroyDnMat(residual); cusparseDestroyDnMat(probe); cusparseDestroyDnMat(z); cusparseDestroyDnMat(projected); cusparseDestroyDnMat(q); cusparseDestroyDnMat(omega); cusparseDestroySpMat(a);
  curandDestroyGenerator(rng); cublasDestroy(blas); cusolverDnDestroy(solver); cusparseDestroy(sparse);
  return 0;
}
