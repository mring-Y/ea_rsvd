#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define CUDA_OK(x) do { const auto e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return false; } } while (0)
#define SPARSE_OK(x) do { const auto e = (x); if (e != CUSPARSE_STATUS_SUCCESS) { std::fprintf(stderr, "cuSPARSE %s:%d: %d\n", __FILE__, __LINE__, static_cast<int>(e)); return false; } } while (0)
#define SOLVER_OK(x) do { const auto e = (x); if (e != CUSOLVER_STATUS_SUCCESS) { std::fprintf(stderr, "cuSOLVER %s:%d: %d\n", __FILE__, __LINE__, static_cast<int>(e)); return false; } } while (0)
#define BLAS_OK(x) do { const auto e = (x); if (e != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "cuBLAS %s:%d: %d\n", __FILE__, __LINE__, static_cast<int>(e)); return false; } } while (0)

struct CsrMatrix { int rows = 0, cols = 0; std::vector<int> offsets, columns; std::vector<float> values; };
#ifndef PRECISION_MODE
#define PRECISION_MODE 0
#endif

static const char* precision_mode_name() {
#if PRECISION_MODE == 1
  return "fp64_monitor";
#elif PRECISION_MODE == 2
  return "kahan_fp32_monitor";
#else
  return "fp32_monitor";
#endif
}

// The synthetic generator matches stage61_dual_lane_dynamic_overlap.cu so that
// the replay uses the same 2048x1536, eight-entry-per-row matrix.
static CsrMatrix synthetic_matrix(int rows = 2048, int cols = 1536, int entries_per_row = 8) {
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

static double energy_sum(const float* values, size_t count) {
#if PRECISION_MODE == 1
  double sum = 0.0; for (size_t i = 0; i < count; ++i) { const double x = static_cast<double>(values[i]); sum += x * x; } return sum;
#elif PRECISION_MODE == 2
  float sum = 0.0F, compensation = 0.0F; for (size_t i = 0; i < count; ++i) { const float x = values[i] * values[i]; const float y = x - compensation; const float t = sum + y; compensation = (t - sum) - y; sum = t; } return static_cast<double>(sum);
#else
  float sum = 0.0F; for (size_t i = 0; i < count; ++i) sum += values[i] * values[i]; return static_cast<double>(sum);
#endif
}

static double energy_sum(const std::vector<float>& values) { return energy_sum(values.data(), values.size()); }
struct CudaCsr { int* offsets = nullptr; int* columns = nullptr; float* values = nullptr; cusparseSpMatDescr_t desc = nullptr; };
struct QrWorkspace { int max_rows = 0, max_cols = 0, work_size = 0; float* tau = nullptr; float* work = nullptr; int* info = nullptr; };
// One resident controller state.  The probe residual is retained for backwards
// compatible traces; the controller decisions below use exact_rho_l/rho_k,
// computed from the same resident Q and the compile-selected monitor policy.
struct StateMetric {
  int width = 0, power = 0;
  double transition_ms = 0.0, monitor_ms = 0.0, cumulative_ms = 0.0;
  float pve = std::numeric_limits<float>::infinity();
  float probe_residual = 0.0F;
  float exact_rho_l = std::numeric_limits<float>::quiet_NaN();
  float exact_rho_k = std::numeric_limits<float>::quiet_NaN();
  float q_orth_error = 0.0F;
  std::string action, reason;
};

static bool read_mtx(const std::string& path, CsrMatrix* out) {
  std::ifstream in(path); std::string line;
  if (!in || !std::getline(in, line) || line.rfind("%%MatrixMarket matrix coordinate", 0) != 0) return false;
  const bool pattern = line.find("pattern") != std::string::npos;
  do { if (!std::getline(in, line)) return false; } while (line.empty() || line[0] == '%');
  int nnz = 0; std::istringstream shape(line); if (!(shape >> out->rows >> out->cols >> nnz)) return false;
  std::vector<std::vector<std::pair<int, float>>> rows(out->rows);
  for (int i = 0; i < nnz; ++i) {
    int r = 0, c = 0; float v = 1.0F; if (!(in >> r >> c)) return false; if (!pattern && !(in >> v)) return false;
    if (r < 1 || r > out->rows || c < 1 || c > out->cols) return false; rows[r - 1].push_back({c - 1, v});
  }
  out->offsets.assign(out->rows + 1, 0); out->columns.reserve(nnz); out->values.reserve(nnz);
  for (int r = 0; r < out->rows; ++r) { out->offsets[r + 1] = out->offsets[r] + static_cast<int>(rows[r].size()); for (auto e : rows[r]) { out->columns.push_back(e.first); out->values.push_back(e.second); } }
  return true;
}

static bool upload(const CsrMatrix& h, CudaCsr* d) {
  CUDA_OK(cudaMalloc(&d->offsets, sizeof(int) * h.offsets.size())); CUDA_OK(cudaMalloc(&d->columns, sizeof(int) * h.columns.size())); CUDA_OK(cudaMalloc(&d->values, sizeof(float) * h.values.size()));
  CUDA_OK(cudaMemcpy(d->offsets, h.offsets.data(), sizeof(int) * h.offsets.size(), cudaMemcpyHostToDevice)); CUDA_OK(cudaMemcpy(d->columns, h.columns.data(), sizeof(int) * h.columns.size(), cudaMemcpyHostToDevice)); CUDA_OK(cudaMemcpy(d->values, h.values.data(), sizeof(float) * h.values.size(), cudaMemcpyHostToDevice));
  SPARSE_OK(cusparseCreateCsr(&d->desc, h.rows, h.cols, static_cast<int64_t>(h.values.size()), d->offsets, d->columns, d->values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F)); return true;
}
static void release(CudaCsr* d) { if (d->desc) cusparseDestroySpMat(d->desc); cudaFree(d->values); cudaFree(d->columns); cudaFree(d->offsets); *d = {}; }

static bool init_qr(cusolverDnHandle_t solver, int rows, int cols, float* scratch, QrWorkspace* qr) {
  qr->max_rows = rows; qr->max_cols = cols; CUDA_OK(cudaMalloc(&qr->tau, sizeof(float) * cols)); CUDA_OK(cudaMalloc(&qr->info, sizeof(int)));
  int g = 0, o = 0; SOLVER_OK(cusolverDnSgeqrf_bufferSize(solver, rows, cols, scratch, rows, &g)); SOLVER_OK(cusolverDnSorgqr_bufferSize(solver, rows, cols, cols, scratch, rows, qr->tau, &o)); qr->work_size = std::max(g, o); CUDA_OK(cudaMalloc(&qr->work, sizeof(float) * qr->work_size)); return true;
}
static void release_qr(QrWorkspace* qr) { cudaFree(qr->work); cudaFree(qr->tau); cudaFree(qr->info); *qr = {}; }
static bool qr(cusolverDnHandle_t solver, float* x, int rows, int cols, QrWorkspace& w) {
  SOLVER_OK(cusolverDnSgeqrf(solver, rows, cols, x, rows, w.tau, w.work, w.work_size, w.info)); SOLVER_OK(cusolverDnSorgqr(solver, rows, cols, cols, x, rows, w.tau, w.work, w.work_size, w.info)); CUDA_OK(cudaDeviceSynchronize()); int info = -1; CUDA_OK(cudaMemcpy(&info, w.info, sizeof(int), cudaMemcpyDeviceToHost)); return info == 0;
}

static bool spmm(cusparseHandle_t sparse, cusparseSpMatDescr_t a, cusparseOperation_t op, int x_rows, int x_cols, const float* x, int y_rows, float* y) {
  cusparseDnMatDescr_t xd = nullptr, yd = nullptr; SPARSE_OK(cusparseCreateDnMat(&xd, x_rows, x_cols, x_rows, const_cast<float*>(x), CUDA_R_32F, CUSPARSE_ORDER_COL)); SPARSE_OK(cusparseCreateDnMat(&yd, y_rows, x_cols, y_rows, y, CUDA_R_32F, CUSPARSE_ORDER_COL));
  const float one = 1.0F, zero = 0.0F; size_t bytes = 0; SPARSE_OK(cusparseSpMM_bufferSize(sparse, op, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, a, xd, &zero, yd, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bytes)); void* buf = nullptr; CUDA_OK(cudaMalloc(&buf, bytes)); SPARSE_OK(cusparseSpMM(sparse, op, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, a, xd, &zero, yd, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buf)); CUDA_OK(cudaDeviceSynchronize()); cudaFree(buf); cusparseDestroyDnMat(yd); cusparseDestroyDnMat(xd); return true;
}
static bool power_step(cusparseHandle_t sparse, cusolverDnHandle_t solver, cusparseSpMatDescr_t a, int m, int n, int l, float* q_in, float* z, float* q_out, QrWorkspace& qr_n, QrWorkspace& qr_m) {
  if (!spmm(sparse, a, CUSPARSE_OPERATION_TRANSPOSE, m, l, q_in, n, z)) return false;
  if (!qr(solver, z, n, l, qr_n)) return false;
  if (!spmm(sparse, a, CUSPARSE_OPERATION_NON_TRANSPOSE, n, l, z, m, q_out)) return false;
  if (!qr(solver, q_out, m, l, qr_m)) return false;
  return true;
}

static bool monitor(cusparseHandle_t sparse, cublasHandle_t blas, cusolverDnHandle_t solver, cusparseSpMatDescr_t a, int m, int n, int l, int rank, float* q, float* probe, float* residual, float* coeff, float* bt, float* sigma, float* svd_work, int svd_work_size, int* info, double matrix_energy, float* out_residual, float* out_orth, const float* previous_sigma, float* out_pve, float* out_rho_l, float* out_rho_k) {
  constexpr int probes = 16;
  if (l < rank || matrix_energy <= 0.0 || !std::isfinite(matrix_energy)) {
    std::fprintf(stderr, "monitor: invalid dimensions or matrix energy (l=%d rank=%d energy=%.9g)\n", l, rank, matrix_energy);
    return false;
  }
  if (!spmm(sparse, a, CUSPARSE_OPERATION_NON_TRANSPOSE, n, probes, probe, m, residual)) return false;
  const float one = 1.0F, minus = -1.0F, zero = 0.0F; BLAS_OK(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, l, probes, m, &one, q, m, residual, m, &zero, coeff, l)); BLAS_OK(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, m, probes, l, &minus, q, m, coeff, l, &one, residual, m));
  std::vector<float> hres(static_cast<size_t>(m) * probes); CUDA_OK(cudaMemcpy(hres.data(), residual, sizeof(float) * hres.size(), cudaMemcpyDeviceToHost)); const double sum = energy_sum(hres); *out_residual = static_cast<float>(std::sqrt(sum / (probes * matrix_energy)));
  std::vector<float> hq(static_cast<size_t>(m) * l); CUDA_OK(cudaMemcpy(hq.data(), q, sizeof(float) * hq.size(), cudaMemcpyDeviceToHost)); float oe = 0.0F; for (int i = 0; i < l; ++i) for (int j = 0; j < l; ++j) { double dot = 0.0; for (int r = 0; r < m; ++r) dot += static_cast<double>(hq[r + i * m]) * hq[r + j * m]; oe = std::max(oe, static_cast<float>(std::fabs(dot - (i == j ? 1.0 : 0.0)))); } *out_orth = oe;
  if (!spmm(sparse, a, CUSPARSE_OPERATION_TRANSPOSE, m, l, q, n, bt)) return false;
  // Keep the energy identity separate from the SVD workspace.  The next
  // solver call overwrites B^T=A^T Q, so copy it first and accumulate in
  // FP64 on the host.  This is the exact residual used by the state policy,
  // rather than the 16-probe monitor retained for compatibility.
  std::vector<float> hbt(static_cast<size_t>(n) * l);
  CUDA_OK(cudaMemcpy(hbt.data(), bt, sizeof(float) * hbt.size(), cudaMemcpyDeviceToHost));
  const double projected_energy = energy_sum(hbt);
  *out_rho_l = static_cast<float>(std::sqrt(std::max(0.0, matrix_energy - projected_energy) / matrix_energy));
  // Sgesvd overwrites B^T in place.  Synchronize and copy devInfo before any
  // host-side interpretation so a failed solver call cannot cascade into an
  // invalid host/device access later in the controller.
  SOLVER_OK(cusolverDnSgesvd(solver, 'N', 'N', n, l, bt, n, sigma, nullptr, n, nullptr, l, svd_work, svd_work_size, nullptr, info));
  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaDeviceSynchronize());
  int host_info = -1;
  CUDA_OK(cudaMemcpy(&host_info, info, sizeof(int), cudaMemcpyDeviceToHost));
  if (host_info != 0) {
    std::fprintf(stderr, "monitor: Sgesvd devInfo=%d (n=%d l=%d)\n", host_info, n, l);
    return false;
  }
  std::vector<float> host_sigma(l); CUDA_OK(cudaMemcpy(host_sigma.data(), sigma, sizeof(float) * l, cudaMemcpyDeviceToHost));
  const double rank_energy = energy_sum(host_sigma.data(), static_cast<size_t>(rank));
  *out_rho_k = static_cast<float>(std::sqrt(std::max(0.0, matrix_energy - rank_energy) / matrix_energy));
  *out_pve = std::numeric_limits<float>::infinity();
  if (previous_sigma) { *out_pve = 0.0F; for (int i = 0; i < rank; ++i) *out_pve = std::max(*out_pve, std::fabs(host_sigma[i] - previous_sigma[i]) / std::max(host_sigma[rank - 1], 1e-7F)); }
  return true;
}

static bool widen(cusparseHandle_t sparse, cublasHandle_t blas, cusolverDnHandle_t solver, cusparseSpMatDescr_t a, int m, int n, int old_l, int new_l, float* omega, float* q, QrWorkspace& qr_m, float* r, float* tau_q, float* work_q, int work_size_q, int* info_q) {
  const int delta = new_l - old_l; float* ext = q + static_cast<size_t>(old_l) * m; const float one = 1.0F, zero = 0.0F, minus = -1.0F; if (!spmm(sparse, a, CUSPARSE_OPERATION_NON_TRANSPOSE, n, delta, omega + static_cast<size_t>(old_l) * n, m, ext)) return false;
  BLAS_OK(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, old_l, delta, m, &one, q, m, ext, m, &zero, r, old_l)); BLAS_OK(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, m, delta, old_l, &minus, q, m, r, old_l, &one, ext, m));
  SOLVER_OK(cusolverDnSgeqrf(solver, m, delta, ext, m, tau_q, work_q, work_size_q, info_q)); SOLVER_OK(cusolverDnSorgqr(solver, m, delta, delta, ext, m, tau_q, work_q, work_size_q, info_q)); CUDA_OK(cudaDeviceSynchronize()); int info = -1; CUDA_OK(cudaMemcpy(&info, info_q, sizeof(int), cudaMemcpyDeviceToHost)); return info == 0;
}

int main(int argc, char** argv) {
  if (argc < 2 || argc > 5) { std::fprintf(stderr, "Usage: incremental_controller matrix.mtx [trace.csv] [seed] [max_width]\n"); return 64; }
  const std::string input = argv[1], csv = argc >= 3 ? argv[2] : "incremental_controller_trace.csv"; const unsigned long long seed = argc >= 4 ? std::strtoull(argv[3], nullptr, 10) : 20260730ULL; const int max_l = argc >= 5 ? std::atoi(argv[4]) : 40; const int rank = 16, min_l = 24, max_q = 2, probes = 16; const float tau_rho_k = 0.20F, tau_rho_l = 0.20F, tau_delta_sigma = 0.01F;
  CsrMatrix h; if ((input == "synthetic" ? (h = synthetic_matrix(), true) : read_mtx(input, &h)) == false || max_l < min_l || max_l > std::min(h.rows, h.cols) || ((max_l - min_l) % 8) != 0) return 65; const double matrix_energy = energy_sum(h.values);
  if (h.rows <= 0 || h.cols <= 0 || h.values.empty() || !std::isfinite(matrix_energy) || matrix_energy <= 0.0) {
    std::fprintf(stderr, "invalid Matrix Market input or zero matrix energy\n");
    return 65;
  }
  std::mt19937_64 gen(seed); std::normal_distribution<float> normal(0.0F, 1.0F); std::vector<float> omega(static_cast<size_t>(h.cols) * max_l), probe(static_cast<size_t>(h.cols) * probes); for (float& v : omega) v = normal(gen); for (float& v : probe) v = normal(gen);
  cusparseHandle_t sparse = nullptr; cusolverDnHandle_t solver = nullptr; cublasHandle_t blas = nullptr; CudaCsr d_a; if (!upload(h, &d_a)) return 1; SPARSE_OK(cusparseCreate(&sparse)); SOLVER_OK(cusolverDnCreate(&solver)); BLAS_OK(cublasCreate(&blas));
  float *d_omega = nullptr, *d_q = nullptr, *d_z = nullptr, *d_probe = nullptr, *d_residual = nullptr, *d_coeff = nullptr, *d_bt = nullptr, *d_sigma = nullptr, *d_sigma_prev = nullptr, *d_r = nullptr, *d_qscratch = nullptr; CUDA_OK(cudaMalloc(&d_omega, sizeof(float) * static_cast<size_t>(h.cols) * max_l)); CUDA_OK(cudaMalloc(&d_q, sizeof(float) * static_cast<size_t>(h.rows) * max_l)); CUDA_OK(cudaMalloc(&d_z, sizeof(float) * static_cast<size_t>(h.cols) * max_l)); CUDA_OK(cudaMalloc(&d_probe, sizeof(float) * static_cast<size_t>(h.cols) * probes)); CUDA_OK(cudaMalloc(&d_residual, sizeof(float) * static_cast<size_t>(h.rows) * probes)); CUDA_OK(cudaMalloc(&d_coeff, sizeof(float) * static_cast<size_t>(max_l) * probes)); CUDA_OK(cudaMalloc(&d_bt, sizeof(float) * static_cast<size_t>(h.cols) * max_l)); CUDA_OK(cudaMalloc(&d_sigma, sizeof(float) * max_l)); CUDA_OK(cudaMalloc(&d_sigma_prev, sizeof(float) * max_l)); CUDA_OK(cudaMalloc(&d_r, sizeof(float) * static_cast<size_t>(max_l) * max_l)); CUDA_OK(cudaMalloc(&d_qscratch, sizeof(float) * static_cast<size_t>(h.rows) * max_l)); CUDA_OK(cudaMemcpy(d_omega, omega.data(), sizeof(float) * omega.size(), cudaMemcpyHostToDevice)); CUDA_OK(cudaMemcpy(d_probe, probe.data(), sizeof(float) * probe.size(), cudaMemcpyHostToDevice));
  QrWorkspace qr_m{}, qr_n{}; if (!init_qr(solver, h.rows, max_l, d_q, &qr_m) || !init_qr(solver, h.cols, max_l, d_z, &qr_n)) return 1; int svd_work_size = 0; SOLVER_OK(cusolverDnSgesvd_bufferSize(solver, h.cols, max_l, &svd_work_size)); float* d_svd_work = nullptr; CUDA_OK(cudaMalloc(&d_svd_work, sizeof(float) * svd_work_size)); int* d_info = nullptr; CUDA_OK(cudaMalloc(&d_info, sizeof(int)));
  // Warm up the library before recording the state path.
  if (!spmm(sparse, d_a.desc, CUSPARSE_OPERATION_NON_TRANSPOSE, h.cols, min_l, d_omega, h.rows, d_q) || !qr(solver, d_q, h.rows, min_l, qr_m)) return 1;
  std::vector<StateMetric> states; int current_l = min_l, current_q = 0; bool have_sigma = false; float last_pve = std::numeric_limits<float>::infinity(); double cumulative = 0.0; std::vector<float> previous_sigma(max_l, 0.0F);
  while (true) {
    StateMetric s; s.width = current_l; s.power = current_q; auto t0 = std::chrono::steady_clock::now();
    if (states.empty()) { if (!spmm(sparse, d_a.desc, CUSPARSE_OPERATION_NON_TRANSPOSE, h.cols, current_l, d_omega, h.rows, d_q) || !qr(solver, d_q, h.rows, current_l, qr_m)) return 1; }
    else if (states.back().action == "add_power_step") { if (!power_step(sparse, solver, d_a.desc, h.rows, h.cols, current_l, d_q, d_z, d_qscratch, qr_n, qr_m)) return 1; CUDA_OK(cudaMemcpy(d_q, d_qscratch, sizeof(float) * static_cast<size_t>(h.rows) * current_l, cudaMemcpyDeviceToDevice)); }
    else if (states.back().action == "expand_sketch") { if (!widen(sparse, blas, solver, d_a.desc, h.rows, h.cols, current_l, current_l + 8, d_omega, d_q, qr_m, d_r, qr_m.tau, qr_m.work, qr_m.work_size, qr_m.info)) return 1; current_l += 8; current_q = 0; s.width = current_l; s.power = current_q; have_sigma = false; last_pve = std::numeric_limits<float>::infinity(); }
    s.transition_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    auto tm = std::chrono::steady_clock::now(); float residual = 0.0F, oe = 0.0F, pve = std::numeric_limits<float>::infinity(), rho_l = std::numeric_limits<float>::quiet_NaN(), rho_k = std::numeric_limits<float>::quiet_NaN(); const float* prev = have_sigma ? previous_sigma.data() : nullptr; if (!monitor(sparse, blas, solver, d_a.desc, h.rows, h.cols, current_l, rank, d_q, d_probe, d_residual, d_coeff, d_bt, d_sigma, d_svd_work, svd_work_size, d_info, matrix_energy, &residual, &oe, prev, &pve, &rho_l, &rho_k)) return 1; s.monitor_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tm).count(); s.probe_residual = residual; s.exact_rho_l = rho_l; s.exact_rho_k = rho_k; s.q_orth_error = oe; s.pve = pve; CUDA_OK(cudaMemcpy(previous_sigma.data(), d_sigma, sizeof(float) * max_l, cudaMemcpyDeviceToHost)); have_sigma = true; if (std::isfinite(pve)) last_pve = pve;
    cumulative += s.transition_ms + s.monitor_ms; s.cumulative_ms = cumulative;
    // Algorithm 1 (exact controller): certify the requested rank first;
    // otherwise force q=1 from the initial q=0 state, refine power while the
    // singular-value change is large, then widen only when rho_l is still high.
    if (rho_k <= tau_rho_k) { s.action = "stop"; s.reason = "exact rho_k below threshold"; states.push_back(s); break; }
    if (current_q == 0 && current_q < max_q) { ++current_q; s.action = "add_power_step"; s.reason = "q=0 requires one power step"; states.push_back(s); continue; }
    if (current_q < max_q && (!std::isfinite(pve) || pve > tau_delta_sigma)) { ++current_q; s.action = "add_power_step"; s.reason = "delta_sigma remains above threshold"; states.push_back(s); continue; }
    if (rho_l > tau_rho_l && current_l < max_l) { s.action = "expand_sketch"; s.reason = "exact rho_l remains above threshold"; states.push_back(s); continue; }
    s.action = "stop_budget"; s.reason = "exact rho_k above threshold and configured q/l budget exhausted"; states.push_back(s); break;
  }
  std::filesystem::path parent = std::filesystem::path(csv).parent_path(); if (!parent.empty()) std::filesystem::create_directories(parent); const bool fresh = !std::filesystem::exists(csv); std::ofstream out(csv, std::ios::app); if (fresh) out << "precision_mode,input,rows,cols,nnz,seed,step,sketch_width,power_steps,transition_ms,monitor_ms,cumulative_ms,pve,probe_residual,q_orthogonality_max_error,action,reason,exact_rho_l,exact_rho_k,delta_sigma\n"; for (size_t i = 0; i < states.size(); ++i) out << precision_mode_name() << ',' << input << ',' << h.rows << ',' << h.cols << ',' << h.values.size() << ',' << seed << ',' << i << ',' << states[i].width << ',' << states[i].power << ',' << states[i].transition_ms << ',' << states[i].monitor_ms << ',' << states[i].cumulative_ms << ',' << states[i].pve << ',' << states[i].probe_residual << ',' << states[i].q_orth_error << ',' << states[i].action << ',' << states[i].reason << ',' << states[i].exact_rho_l << ',' << states[i].exact_rho_k << ',' << states[i].pve << '\n';
  std::printf("precision=%s incremental controller %s seed=%llu: states=%zu terminal=(l=%d,q=%d) cumulative=%.3f ms action=%s\n", precision_mode_name(), input.c_str(), seed, states.size(), states.back().width, states.back().power, states.back().cumulative_ms, states.back().action.c_str());
  release_qr(&qr_n); release_qr(&qr_m); cudaFree(d_info); cudaFree(d_svd_work); cudaFree(d_qscratch); cudaFree(d_r); cudaFree(d_sigma_prev); cudaFree(d_sigma); cudaFree(d_bt); cudaFree(d_coeff); cudaFree(d_residual); cudaFree(d_probe); cudaFree(d_z); cudaFree(d_q); cudaFree(d_omega); release(&d_a); cublasDestroy(blas); cusolverDnDestroy(solver); cusparseDestroy(sparse); return 0;
}
