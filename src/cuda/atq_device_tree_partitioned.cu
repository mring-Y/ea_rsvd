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
#include <numeric>
#include <thread>
#include <vector>

// This file is an independent engineering prototype.  It computes A^T Q in
// column partitions, reuses a fixed pool of partial buffers, and reduces each
// pool group with a device-side binary tree before updating its B partition.

struct CsrMatrix {
  int rows = 0, cols = 0;
  std::vector<int> offsets, columns;
  std::vector<float> values;
};

static bool cuda_ok(cudaError_t e, const char* file, int line) {
  if (e == cudaSuccess) return true;
  std::fprintf(stderr, "CUDA %s:%d: %s\n", file, line, cudaGetErrorString(e));
  return false;
}
static bool sparse_ok(cusparseStatus_t e, const char* file, int line) {
  if (e == CUSPARSE_STATUS_SUCCESS) return true;
  std::fprintf(stderr, "cuSPARSE %s:%d: %d\n", file, line, static_cast<int>(e));
  return false;
}
static bool solver_ok(cusolverStatus_t e, const char* file, int line) {
  if (e == CUSOLVER_STATUS_SUCCESS) return true;
  std::fprintf(stderr, "cuSOLVER %s:%d: %d\n", file, line, static_cast<int>(e)); return false;
}
#define CUDA_MAIN(x) do { if (!cuda_ok((x), __FILE__, __LINE__)) return 1; } while (0)
#define SPARSE_MAIN(x) do { if (!sparse_ok((x), __FILE__, __LINE__)) return 1; } while (0)
#define SOLVER_MAIN(x) do { if (!solver_ok((x), __FILE__, __LINE__)) return 1; } while (0)

static bool read_mtx(const std::string& path, CsrMatrix* out) {
  std::ifstream in(path); std::string line;
  if (!in || !std::getline(in, line) || line.rfind("%%MatrixMarket matrix coordinate", 0) != 0) return false;
  const bool pattern = line.find("pattern") != std::string::npos;
  do { if (!std::getline(in, line)) return false; } while (line.empty() || line[0] == '%');
  int nnz = 0; std::istringstream shape(line);
  if (!(shape >> out->rows >> out->cols >> nnz)) return false;
  std::vector<std::vector<std::pair<int, float>>> rows(out->rows);
  for (int i = 0; i < nnz; ++i) {
    int r = 0, c = 0; float v = 1.0F;
    if (!(in >> r >> c) || r < 1 || r > out->rows || c < 1 || c > out->cols) return false;
    if (!pattern && !(in >> v)) return false;
    rows[r - 1].push_back({c - 1, v});
  }
  out->offsets.assign(out->rows + 1, 0);
  out->columns.reserve(nnz); out->values.reserve(nnz);
  for (int r = 0; r < out->rows; ++r) {
    out->offsets[r + 1] = out->offsets[r] + static_cast<int>(rows[r].size());
    for (const auto& x : rows[r]) { out->columns.push_back(x.first); out->values.push_back(x.second); }
  }
  return true;
}

static CsrMatrix synthetic(int rows, int cols, int entries_per_row) {
  CsrMatrix a; a.rows = rows; a.cols = cols; a.offsets.assign(rows + 1, 0);
  std::mt19937 gen(20260928); std::uniform_int_distribution<int> col(0, cols - 1);
  std::uniform_real_distribution<float> val(-1.0F, 1.0F);
  for (int r = 0; r < rows; ++r) {
    std::vector<int> chosen; chosen.reserve(entries_per_row);
    while (static_cast<int>(chosen.size()) < entries_per_row) {
      int c = col(gen);
      if (std::find(chosen.begin(), chosen.end(), c) == chosen.end()) chosen.push_back(c);
    }
    std::sort(chosen.begin(), chosen.end());
    for (int c : chosen) { a.columns.push_back(c); a.values.push_back(val(gen)); }
    a.offsets[r + 1] = static_cast<int>(a.columns.size());
  }
  return a;
}

// Pairwise tree level.  The destination is the first element of every pair;
// this permits non-power-of-two group sizes without a temporary allocation.
__global__ static void tree_level(float* pool, size_t leaf_stride, int stride, int active, size_t n) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  const int base = pair * 2 * stride;
  if (base >= active || base + stride >= active) return;
  float* dst = pool + static_cast<size_t>(base) * leaf_stride;
  const float* src = pool + static_cast<size_t>(base + stride) * leaf_stride;
  for (size_t i = blockIdx.y * blockDim.y + threadIdx.y; i < n; i += blockDim.y * gridDim.y) dst[i] += src[i];
}

// Dynamic scheduling uses a compact device index list because completed slots
// can belong to arbitrary workers.  The destination/source leaves remain in
// the fixed pool, so no temporary host reduction or slot copy is required.
__global__ static void tree_level_indexed(float* pool, size_t leaf_stride, const int* indices,
                                          int stride, int active, size_t n) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  const int base = pair * 2 * stride;
  if (base >= active || base + stride >= active) return;
  const int dst_index = indices[base];
  const int src_index = indices[base + stride];
  float* dst = pool + static_cast<size_t>(dst_index) * leaf_stride;
  const float* src = pool + static_cast<size_t>(src_index) * leaf_stride;
  for (size_t i = blockIdx.y * blockDim.y + threadIdx.y; i < n; i += blockDim.y * gridDim.y)
    dst[i] += src[i];
}

__global__ static void add_to_b(const float* src, float* b, int b_rows, int pstart, int pcols, int l) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t n = static_cast<size_t>(pcols) * l;
  if (i < n) {
    const int local_col = static_cast<int>(i % pcols);
    const int c = static_cast<int>(i / pcols);
    b[pstart + local_col + static_cast<size_t>(c) * b_rows] += src[i];
  }
}

struct Slot {
  cudaStream_t stream = nullptr;
  cusparseHandle_t sparse = nullptr;
  cudaEvent_t ready = nullptr, free_event = nullptr;
  // Descriptors remain owned by the slot until its stream work has completed.
  // Destroying them immediately after cusparseSpMM submission is unsafe on
  // toolkits that defer descriptor reads until the operation executes.
  cusparseSpMatDescr_t mat_desc = nullptr;
  cusparseDnMatDescr_t q_desc = nullptr, b_desc = nullptr;
  int* h_offsets = nullptr; int* h_columns = nullptr; float* h_values = nullptr; float* h_q = nullptr;
  int* d_offsets = nullptr; int* d_columns = nullptr; float* d_values = nullptr; float* d_q = nullptr;
  void* d_buffer = nullptr; size_t buffer_bytes = 0;
  size_t pinned_bytes = 0, device_bytes = 0;
  int dynamic_state = 0;  // 0=free, 1=compute pending, 2=reduce pending
  int dynamic_task = -1, dynamic_partition = -1;
};

static void destroy_slot(Slot* s) {
  if (s->mat_desc) cusparseDestroySpMat(s->mat_desc);
  if (s->q_desc) cusparseDestroyDnMat(s->q_desc);
  if (s->b_desc) cusparseDestroyDnMat(s->b_desc);
  if (s->ready) cudaEventDestroy(s->ready); if (s->free_event) cudaEventDestroy(s->free_event);
  if (s->sparse) cusparseDestroy(s->sparse);
  if (s->stream) cudaStreamDestroy(s->stream);
  cudaFreeHost(s->h_offsets); cudaFreeHost(s->h_columns); cudaFreeHost(s->h_values); cudaFreeHost(s->h_q);
  cudaFree(s->d_offsets); cudaFree(s->d_columns); cudaFree(s->d_values); cudaFree(s->d_q); cudaFree(s->d_buffer);
  *s = {};
}

int main(int argc, char** argv) {
  const std::string input = argc >= 2 ? argv[1] : "synthetic";
  const std::string output = argc >= 3 ? argv[2] : "atq_device_tree_partitioned.csv";
  const int block_rows = argc >= 4 ? std::atoi(argv[3]) : 257;
  const int rank = argc >= 5 ? std::atoi(argv[4]) : 16;
  const int partitions = argc >= 6 ? std::atoi(argv[5]) : 4;
  const int pool_slots = argc >= 7 ? std::atoi(argv[6]) : 4;
  const bool integrated = argc >= 8 ? std::atoi(argv[7]) != 0 : false;
  const bool dynamic_mode = argc >= 9 ? std::atoi(argv[8]) != 0 : false;
  if (!cuda_ok(cudaSetDevice(0), __FILE__, __LINE__)) return 1;
  constexpr int oversampling = 8;
  CsrMatrix a;
  if (input == "synthetic") a = synthetic(2048, 1536, 8);
  else if (!read_mtx(input, &a)) { std::fprintf(stderr, "Cannot read Matrix Market input: %s\n", input.c_str()); return 65; }
  const int l = rank + oversampling;
  if (block_rows <= 0 || rank <= 0 || partitions <= 0 || pool_slots <= 0 || l > std::min(a.rows, a.cols)) return 65;
  const int tasks = (a.rows + block_rows - 1) / block_rows;
  const int max_rows = std::min(block_rows, a.rows);
  int max_nnz = 0;
  for (int t = 0; t < tasks; ++t) {
    int first = t * block_rows, rows = std::min(block_rows, a.rows - first);
    max_nnz = std::max(max_nnz, a.offsets[first + rows] - a.offsets[first]);
  }
  std::mt19937_64 rng(20260928); std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> h_q(static_cast<size_t>(a.rows) * l);
  for (float& x : h_q) x = normal(rng);
  double qr_ms = 0.0, svd_ms = 0.0, q_orthogonality = 0.0;
  int qr_info = 0, svd_info = 0;
  size_t integrated_device_bytes = 0;
  if (integrated) {
    // Generate Q on the GPU through the same A*Omega -> QR path used by the
    // end-to-end lane, then feed that Q into the tree/pool A^TQ stage below.
    int* d_full_offsets = nullptr; int* d_full_columns = nullptr; float* d_full_values = nullptr;
    float* d_omega = nullptr; float* d_y = nullptr; float* d_tau = nullptr; float* d_work = nullptr; int* d_info = nullptr; void* d_spmm = nullptr;
    cusparseHandle_t full_sparse = nullptr; cusolverDnHandle_t full_solver = nullptr; cusparseSpMatDescr_t full_mat = nullptr; cusparseDnMatDescr_t omega_mat = nullptr, y_mat = nullptr;
    CUDA_MAIN(cudaMalloc(&d_full_offsets, sizeof(int) * a.offsets.size())); CUDA_MAIN(cudaMalloc(&d_full_columns, sizeof(int) * a.columns.size())); CUDA_MAIN(cudaMalloc(&d_full_values, sizeof(float) * a.values.size()));
    CUDA_MAIN(cudaMemcpy(d_full_offsets, a.offsets.data(), sizeof(int) * a.offsets.size(), cudaMemcpyHostToDevice)); CUDA_MAIN(cudaMemcpy(d_full_columns, a.columns.data(), sizeof(int) * a.columns.size(), cudaMemcpyHostToDevice)); CUDA_MAIN(cudaMemcpy(d_full_values, a.values.data(), sizeof(float) * a.values.size(), cudaMemcpyHostToDevice));
    CUDA_MAIN(cudaMalloc(&d_omega, sizeof(float) * static_cast<size_t>(a.cols) * l)); CUDA_MAIN(cudaMalloc(&d_y, sizeof(float) * static_cast<size_t>(a.rows) * l)); CUDA_MAIN(cudaMalloc(&d_tau, sizeof(float) * l)); CUDA_MAIN(cudaMalloc(&d_info, sizeof(int)));
    integrated_device_bytes = sizeof(int) * a.offsets.size() + sizeof(int) * a.columns.size() + sizeof(float) * a.values.size() + sizeof(float) * static_cast<size_t>(a.cols) * l + sizeof(float) * static_cast<size_t>(a.rows) * l + sizeof(float) * l + sizeof(int);
    std::vector<float> h_omega(static_cast<size_t>(a.cols) * l); for (float& x : h_omega) x = normal(rng);
    CUDA_MAIN(cudaMemcpy(d_omega, h_omega.data(), sizeof(float) * h_omega.size(), cudaMemcpyHostToDevice));
    SPARSE_MAIN(cusparseCreate(&full_sparse)); SOLVER_MAIN(cusolverDnCreate(&full_solver));
    SPARSE_MAIN(cusparseCreateCsr(&full_mat, a.rows, a.cols, static_cast<int>(a.values.size()), d_full_offsets, d_full_columns, d_full_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    SPARSE_MAIN(cusparseCreateDnMat(&omega_mat, a.cols, l, a.cols, d_omega, CUDA_R_32F, CUSPARSE_ORDER_COL)); SPARSE_MAIN(cusparseCreateDnMat(&y_mat, a.rows, l, a.rows, d_y, CUDA_R_32F, CUSPARSE_ORDER_COL));
    const float one = 1.0F, zero = 0.0F; size_t spmm_bytes = 0; SPARSE_MAIN(cusparseSpMM_bufferSize(full_sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, full_mat, omega_mat, &zero, y_mat, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &spmm_bytes)); CUDA_MAIN(cudaMalloc(&d_spmm, spmm_bytes)); integrated_device_bytes += spmm_bytes;
    const auto qr_start = std::chrono::steady_clock::now(); SPARSE_MAIN(cusparseSpMM(full_sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, full_mat, omega_mat, &zero, y_mat, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_spmm));
    int geqrf_work = 0, orgqr_work = 0; SOLVER_MAIN(cusolverDnSgeqrf_bufferSize(full_solver, a.rows, l, d_y, a.rows, &geqrf_work)); SOLVER_MAIN(cusolverDnSorgqr_bufferSize(full_solver, a.rows, l, l, d_y, a.rows, d_tau, &orgqr_work)); const int qr_work = std::max(geqrf_work, orgqr_work); CUDA_MAIN(cudaMalloc(&d_work, sizeof(float) * qr_work)); integrated_device_bytes += sizeof(float) * qr_work;
    SOLVER_MAIN(cusolverDnSgeqrf(full_solver, a.rows, l, d_y, a.rows, d_tau, d_work, qr_work, d_info)); SOLVER_MAIN(cusolverDnSorgqr(full_solver, a.rows, l, l, d_y, a.rows, d_tau, d_work, qr_work, d_info)); CUDA_MAIN(cudaDeviceSynchronize()); qr_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - qr_start).count();
    CUDA_MAIN(cudaMemcpy(h_q.data(), d_y, sizeof(float) * h_q.size(), cudaMemcpyDeviceToHost)); CUDA_MAIN(cudaMemcpy(&qr_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    for (int c1 = 0; c1 < l; ++c1) for (int c2 = 0; c2 < l; ++c2) { double dot = 0.0; for (int r = 0; r < a.rows; ++r) dot += static_cast<double>(h_q[r + static_cast<size_t>(c1) * a.rows]) * h_q[r + static_cast<size_t>(c2) * a.rows]; q_orthogonality = std::max(q_orthogonality, std::fabs(dot - (c1 == c2 ? 1.0 : 0.0))); }
    cusparseDestroyDnMat(y_mat); cusparseDestroyDnMat(omega_mat); cusparseDestroySpMat(full_mat); cusolverDnDestroy(full_solver); cusparseDestroy(full_sparse); cudaFree(d_spmm); cudaFree(d_work); cudaFree(d_info); cudaFree(d_tau); cudaFree(d_y); cudaFree(d_omega); cudaFree(d_full_values); cudaFree(d_full_columns); cudaFree(d_full_offsets);
  }
  const int max_part_cols = (a.cols + partitions - 1) / partitions;
  const size_t leaf_elements = static_cast<size_t>(max_part_cols) * l;
  size_t free_before = 0, total_mem = 0; CUDA_MAIN(cudaMemGetInfo(&free_before, &total_mem));
  std::vector<Slot> slots(pool_slots);
  size_t tracked_device = 0, pinned_total = 0;
  for (Slot& s : slots) {
    CUDA_MAIN(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
    SPARSE_MAIN(cusparseCreate(&s.sparse)); SPARSE_MAIN(cusparseSetStream(s.sparse, s.stream));
    CUDA_MAIN(cudaEventCreateWithFlags(&s.ready, cudaEventDisableTiming));
    if (dynamic_mode) CUDA_MAIN(cudaEventCreateWithFlags(&s.free_event, cudaEventDisableTiming));
    CUDA_MAIN(cudaMallocHost(&s.h_offsets, sizeof(int) * (max_rows + 1)));
    CUDA_MAIN(cudaMallocHost(&s.h_columns, sizeof(int) * max_nnz));
    CUDA_MAIN(cudaMallocHost(&s.h_values, sizeof(float) * max_nnz));
    CUDA_MAIN(cudaMallocHost(&s.h_q, sizeof(float) * max_rows * l));
    s.pinned_bytes = sizeof(int) * (max_rows + 1) + sizeof(int) * max_nnz + sizeof(float) * max_nnz + sizeof(float) * max_rows * l;
    pinned_total += s.pinned_bytes;
    CUDA_MAIN(cudaMalloc(&s.d_offsets, sizeof(int) * (max_rows + 1))); s.device_bytes += sizeof(int) * (max_rows + 1);
    CUDA_MAIN(cudaMalloc(&s.d_columns, sizeof(int) * max_nnz)); s.device_bytes += sizeof(int) * max_nnz;
    CUDA_MAIN(cudaMalloc(&s.d_values, sizeof(float) * max_nnz)); s.device_bytes += sizeof(float) * max_nnz;
    CUDA_MAIN(cudaMalloc(&s.d_q, sizeof(float) * max_rows * l)); s.device_bytes += sizeof(float) * max_rows * l;
    tracked_device += s.device_bytes;
  }
  // One fixed pool is reused for every B partition and every task group.
  float* d_pool = nullptr; CUDA_MAIN(cudaMalloc(&d_pool, sizeof(float) * leaf_elements * pool_slots));
  tracked_device += sizeof(float) * leaf_elements * pool_slots;
  int* d_indices = nullptr;
  if (dynamic_mode) {
    // Each slot owns an index-list buffer.  A new reduction may be queued
    // while an older group is still draining; slot-specific storage prevents
    // one group's tree from observing another group's index overwrite.
    CUDA_MAIN(cudaMalloc(&d_indices, sizeof(int) * pool_slots * pool_slots));
    tracked_device += sizeof(int) * pool_slots * pool_slots;
  }
  float* d_b = nullptr; const size_t b_elements = static_cast<size_t>(a.cols) * l;
  CUDA_MAIN(cudaMalloc(&d_b, sizeof(float) * b_elements)); tracked_device += sizeof(float) * b_elements;
  CUDA_MAIN(cudaMemset(d_b, 0, sizeof(float) * b_elements));
  cudaStream_t reduce_stream = nullptr; CUDA_MAIN(cudaStreamCreateWithFlags(&reduce_stream, cudaStreamNonBlocking));
  std::vector<float> h_b(b_elements, 0.0F);
  int completed = 0, polls = 0, forced_sync = 0, tree_groups = 0;
  int cost_sorted_tasks = 0;
  double fifo_load_imbalance = 0.0, lpt_load_imbalance = 0.0;
  const auto wall_start = std::chrono::steady_clock::now();
  bool failed = false;
  if (!dynamic_mode) for (int p = 0; p < partitions && !failed; ++p) {
    const int pstart = (static_cast<long long>(p) * a.cols) / partitions;
    const int pend = (static_cast<long long>(p + 1) * a.cols) / partitions;
    const int pcols = pend - pstart;
    const size_t n = static_cast<size_t>(pcols) * l;
    // Least-processing-time order distributes heterogeneous row blocks while
    // retaining fixed-size pool groups and race-free partition ownership.
    std::vector<int> task_order(tasks), fifo(tasks), costs(tasks, 0);
    std::iota(task_order.begin(), task_order.end(), 0);
    std::iota(fifo.begin(), fifo.end(), 0);
    for (int t = 0; t < tasks; ++t) {
      const int first = t * block_rows, rows = std::min(block_rows, a.rows - first);
      for (int r = first; r < first + rows; ++r)
        for (int j = a.offsets[r]; j < a.offsets[r + 1]; ++j)
          costs[t] += (a.columns[j] >= pstart && a.columns[j] < pend) ? 1 : 0;
    }
    std::sort(task_order.begin(), task_order.end(), [&](int x, int y) { return costs[x] > costs[y]; });
    if (tasks > 1 && costs[task_order.front()] != costs[task_order.back()]) ++cost_sorted_tasks;
    std::vector<std::vector<int>> assigned(pool_slots);
    std::vector<long long> lpt_load(pool_slots, 0), fifo_load(pool_slots, 0);
    for (int t = 0; t < tasks; ++t) fifo_load[t % pool_slots] += costs[t];
    for (int task : task_order) {
      const int worker = static_cast<int>(std::min_element(lpt_load.begin(), lpt_load.end()) - lpt_load.begin());
      assigned[worker].push_back(task);
      lpt_load[worker] += costs[task];
    }
    long long lpt_max = 0, fifo_max = 0, total_partition_cost = 0;
    for (int t = 0; t < tasks; ++t) total_partition_cost += costs[t];
    for (int worker = 0; worker < pool_slots; ++worker) { lpt_max = std::max(lpt_max, lpt_load[worker]); fifo_max = std::max(fifo_max, fifo_load[worker]); }
    const double mean_worker_load = static_cast<double>(total_partition_cost) / pool_slots;
    if (mean_worker_load > 0.0) {
      fifo_load_imbalance = std::max(fifo_load_imbalance, fifo_max / mean_worker_load);
      lpt_load_imbalance = std::max(lpt_load_imbalance, lpt_max / mean_worker_load);
    }
    size_t max_worker_tasks = 0; for (const auto& jobs : assigned) max_worker_tasks = std::max(max_worker_tasks, jobs.size());
    task_order.clear(); task_order.reserve(tasks);
    for (size_t round = 0; round < max_worker_tasks; ++round)
      for (int worker = 0; worker < pool_slots; ++worker)
        if (round < assigned[worker].size()) task_order.push_back(assigned[worker][round]);
    for (int group_first = 0; group_first < tasks && !failed; group_first += pool_slots) {
      const int active = std::min(pool_slots, tasks - group_first);
      // Build and submit at most pool_slots row blocks into the reusable pool.
      for (int k = 0; k < active; ++k) {
        Slot& s = slots[k]; const int task = task_order[group_first + k];
        const int first = task * block_rows, rows = std::min(block_rows, a.rows - first);
        int local_nnz = 0; s.h_offsets[0] = 0;
        for (int r = 0; r < rows; ++r) {
          for (int j = a.offsets[first + r]; j < a.offsets[first + r + 1]; ++j) {
            if (a.columns[j] >= pstart && a.columns[j] < pend) { s.h_columns[local_nnz] = a.columns[j] - pstart; s.h_values[local_nnz] = a.values[j]; ++local_nnz; }
          }
          s.h_offsets[r + 1] = local_nnz;
        }
        for (int c = 0; c < l; ++c) std::copy_n(h_q.data() + first + static_cast<size_t>(c) * a.rows, rows, s.h_q + static_cast<size_t>(c) * rows);
        // Empty row-block/partition tiles are valid for sparse inputs.  Do
        // not pass an nnz=0 CSR descriptor to cuSPARSE: some toolkit
        // versions reject that descriptor even though its mathematical
        // contribution is the zero tile.
        if (local_nnz == 0) {
          if (!cuda_ok(cudaMemsetAsync(d_pool + static_cast<size_t>(k) * leaf_elements,
                                       0, sizeof(float) * leaf_elements, s.stream), __FILE__, __LINE__) ||
              !cuda_ok(cudaEventRecord(s.ready, s.stream), __FILE__, __LINE__)) {
            failed = true;
          }
          continue;
        }
        Slot& slot = s;
        if (slot.mat_desc) { cusparseDestroySpMat(slot.mat_desc); slot.mat_desc = nullptr; }
        if (slot.q_desc) { cusparseDestroyDnMat(slot.q_desc); slot.q_desc = nullptr; }
        if (slot.b_desc) { cusparseDestroyDnMat(slot.b_desc); slot.b_desc = nullptr; }
        cusparseSpMatDescr_t mat = nullptr; cusparseDnMatDescr_t q = nullptr, b = nullptr;
        const float one = 1.0F, zero = 0.0F; size_t bytes = 0;
        if (!sparse_ok(cusparseCreateCsr(&mat, rows, pcols, local_nnz, s.d_offsets, s.d_columns, s.d_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F), __FILE__, __LINE__) ||
            !sparse_ok(cusparseCreateDnMat(&q, rows, l, rows, s.d_q, CUDA_R_32F, CUSPARSE_ORDER_COL), __FILE__, __LINE__) ||
            !sparse_ok(cusparseCreateDnMat(&b, pcols, l, pcols, d_pool + static_cast<size_t>(k) * leaf_elements, CUDA_R_32F, CUSPARSE_ORDER_COL), __FILE__, __LINE__)) { failed = true; break; }
        if (!sparse_ok(cusparseSpMM_bufferSize(s.sparse, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat, q, &zero, b, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bytes), __FILE__, __LINE__)) { failed = true; }
        if (!failed && bytes > s.buffer_bytes) {
          const size_t old = s.buffer_bytes; cudaFree(s.d_buffer); if (!cuda_ok(cudaMalloc(&s.d_buffer, bytes), __FILE__, __LINE__)) failed = true;
          s.buffer_bytes = bytes; s.device_bytes += bytes - old; tracked_device += bytes - old;
        }
        if (!failed && (!cuda_ok(cudaMemcpyAsync(s.d_offsets, s.h_offsets, sizeof(int) * (rows + 1), cudaMemcpyHostToDevice, s.stream), __FILE__, __LINE__) ||
                        !cuda_ok(cudaMemcpyAsync(s.d_columns, s.h_columns, sizeof(int) * local_nnz, cudaMemcpyHostToDevice, s.stream), __FILE__, __LINE__) ||
                        !cuda_ok(cudaMemcpyAsync(s.d_values, s.h_values, sizeof(float) * local_nnz, cudaMemcpyHostToDevice, s.stream), __FILE__, __LINE__) ||
                        !cuda_ok(cudaMemcpyAsync(s.d_q, s.h_q, sizeof(float) * rows * l, cudaMemcpyHostToDevice, s.stream), __FILE__, __LINE__) ||
                        !sparse_ok(cusparseSpMM(s.sparse, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat, q, &zero, b, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, s.d_buffer), __FILE__, __LINE__) ||
                        !cuda_ok(cudaEventRecord(s.ready, s.stream), __FILE__, __LINE__))) failed = true;
        if (failed) {
          if (b) cusparseDestroyDnMat(b); if (q) cusparseDestroyDnMat(q); if (mat) cusparseDestroySpMat(mat);
        } else {
          slot.mat_desc = mat; slot.q_desc = q; slot.b_desc = b;
        }
      }
      if (failed) break;
      // The reduction stream waits on all leaves, then performs binary levels.
      for (int k = 0; k < active; ++k) { if (!cuda_ok(cudaStreamWaitEvent(reduce_stream, slots[k].ready, 0), __FILE__, __LINE__)) { failed = true; break; } }
      if (failed) break;
      if (active < pool_slots &&
          !cuda_ok(cudaMemsetAsync(d_pool + static_cast<size_t>(active) * leaf_elements,
                                   0, sizeof(float) * (pool_slots - active) * leaf_elements,
                                   reduce_stream), __FILE__, __LINE__)) {
        failed = true;
        break;
      }
      for (int stride = 1; stride < active; stride *= 2) {
        const int pairs = (active + 2 * stride - 1) / (2 * stride);
        dim3 block(8, 32); dim3 grid((pairs + block.x - 1) / block.x, 1);
        tree_level<<<grid, block, 0, reduce_stream>>>(d_pool, leaf_elements, stride, active, n);
        if (!cuda_ok(cudaGetLastError(), __FILE__, __LINE__)) { failed = true; break; }
      }
      if (failed) break;
      const int threads = 256; const int blocks = static_cast<int>((n + threads - 1) / threads);
      add_to_b<<<blocks, threads, 0, reduce_stream>>>(d_pool, d_b, a.cols, pstart, pcols, l);
      if (!cuda_ok(cudaGetLastError(), __FILE__, __LINE__)) { failed = true; break; }
      ++tree_groups; completed += active;
      // A group cannot reuse the pool until its final B update is complete.
      if (!cuda_ok(cudaStreamSynchronize(reduce_stream), __FILE__, __LINE__)) { failed = true; break; }
      for (int k = 0; k < active; ++k) {
        Slot& slot = slots[k];
        if (slot.mat_desc) { cusparseDestroySpMat(slot.mat_desc); slot.mat_desc = nullptr; }
        if (slot.q_desc) { cusparseDestroyDnMat(slot.q_desc); slot.q_desc = nullptr; }
        if (slot.b_desc) { cusparseDestroyDnMat(slot.b_desc); slot.b_desc = nullptr; }
      }
      ++polls; if (active == pool_slots) ++forced_sync;
    }
  }
  if (dynamic_mode && !failed) {
    // Completion-event scheduler: every row-block/partition pair is a task.
    // A slot is returned to the queue only after the reduction stream records
    // its free event, so the fixed pool can be reused without a group barrier.
    struct DynamicTask { int block = 0; int partition = 0; int cost = 0; };
    std::vector<DynamicTask> dyn_tasks;
    dyn_tasks.reserve(static_cast<size_t>(tasks) * partitions);
    for (int p = 0; p < partitions; ++p) {
      const int pstart = static_cast<int>((static_cast<long long>(p) * a.cols) / partitions);
      const int pend = static_cast<int>((static_cast<long long>(p + 1) * a.cols) / partitions);
      for (int t = 0; t < tasks; ++t) {
        const int first = t * block_rows, rows = std::min(block_rows, a.rows - first);
        int cost = 0;
        for (int r = first; r < first + rows; ++r)
          for (int j = a.offsets[r]; j < a.offsets[r + 1]; ++j)
            cost += (a.columns[j] >= pstart && a.columns[j] < pend) ? 1 : 0;
        dyn_tasks.push_back({t, p, cost});
      }
    }
    std::stable_sort(dyn_tasks.begin(), dyn_tasks.end(), [](const DynamicTask& x, const DynamicTask& y) { return x.cost > y.cost; });
    if (dyn_tasks.size() > 1 && dyn_tasks.front().cost != dyn_tasks.back().cost) cost_sorted_tasks = 1;
    size_t next_dynamic = 0;
    int dynamic_completed = 0;
    auto submit_dynamic = [&](int slot_index, const DynamicTask& task) -> bool {
      Slot& s = slots[slot_index];
      const int pstart = static_cast<int>((static_cast<long long>(task.partition) * a.cols) / partitions);
      const int pend = static_cast<int>((static_cast<long long>(task.partition + 1) * a.cols) / partitions);
      const int pcols = pend - pstart;
      const int first = task.block * block_rows, rows = std::min(block_rows, a.rows - first);
      int local_nnz = 0; s.h_offsets[0] = 0;
      for (int r = 0; r < rows; ++r) {
        for (int j = a.offsets[first + r]; j < a.offsets[first + r + 1]; ++j) {
          if (a.columns[j] >= pstart && a.columns[j] < pend) {
            s.h_columns[local_nnz] = a.columns[j] - pstart; s.h_values[local_nnz] = a.values[j]; ++local_nnz;
          }
        }
        s.h_offsets[r + 1] = local_nnz;
      }
      for (int c = 0; c < l; ++c)
        std::copy_n(h_q.data() + first + static_cast<size_t>(c) * a.rows, rows, s.h_q + static_cast<size_t>(c) * rows);
      // See the static scheduler above: an empty tile contributes zeros and
      // still participates in the completion-event queue.
      if (local_nnz == 0) {
        if (!cuda_ok(cudaMemsetAsync(d_pool + static_cast<size_t>(slot_index) * leaf_elements,
                                     0, sizeof(float) * leaf_elements, s.stream), __FILE__, __LINE__) ||
            !cuda_ok(cudaEventRecord(s.ready, s.stream), __FILE__, __LINE__)) {
          return false;
        }
        s.dynamic_task = -1; s.dynamic_partition = task.partition; s.dynamic_state = 1;
        return true;
      }
      cusparseSpMatDescr_t mat = nullptr; cusparseDnMatDescr_t q = nullptr, b = nullptr;
      const float one = 1.0F, zero = 0.0F; size_t bytes = 0;
      if (!sparse_ok(cusparseCreateCsr(&mat, rows, pcols, local_nnz, s.d_offsets, s.d_columns, s.d_values,
                                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F), __FILE__, __LINE__) ||
          !sparse_ok(cusparseCreateDnMat(&q, rows, l, rows, s.d_q, CUDA_R_32F, CUSPARSE_ORDER_COL), __FILE__, __LINE__) ||
          !sparse_ok(cusparseCreateDnMat(&b, pcols, l, pcols, d_pool + static_cast<size_t>(slot_index) * leaf_elements,
                                         CUDA_R_32F, CUSPARSE_ORDER_COL), __FILE__, __LINE__)) {
        if (b) cusparseDestroyDnMat(b); if (q) cusparseDestroyDnMat(q); if (mat) cusparseDestroySpMat(mat); return false;
      }
      if (!sparse_ok(cusparseSpMM_bufferSize(s.sparse, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             &one, mat, q, &zero, b, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bytes), __FILE__, __LINE__)) {
        cusparseDestroyDnMat(b); cusparseDestroyDnMat(q); cusparseDestroySpMat(mat); return false;
      }
      if (bytes > s.buffer_bytes) {
        const size_t old = s.buffer_bytes; cudaFree(s.d_buffer);
        if (!cuda_ok(cudaMalloc(&s.d_buffer, bytes), __FILE__, __LINE__)) {
          cusparseDestroyDnMat(b); cusparseDestroyDnMat(q); cusparseDestroySpMat(mat); return false;
        }
        s.buffer_bytes = bytes; s.device_bytes += bytes - old; tracked_device += bytes - old;
      }
      const bool copies_ok = cuda_ok(cudaMemcpyAsync(s.d_offsets, s.h_offsets, sizeof(int) * (rows + 1), cudaMemcpyHostToDevice, s.stream), __FILE__, __LINE__) &&
        cuda_ok(cudaMemcpyAsync(s.d_columns, s.h_columns, sizeof(int) * local_nnz, cudaMemcpyHostToDevice, s.stream), __FILE__, __LINE__) &&
        cuda_ok(cudaMemcpyAsync(s.d_values, s.h_values, sizeof(float) * local_nnz, cudaMemcpyHostToDevice, s.stream), __FILE__, __LINE__) &&
        cuda_ok(cudaMemcpyAsync(s.d_q, s.h_q, sizeof(float) * rows * l, cudaMemcpyHostToDevice, s.stream), __FILE__, __LINE__);
      bool ok = copies_ok && sparse_ok(cusparseSpMM(s.sparse, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                     &one, mat, q, &zero, b, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, s.d_buffer), __FILE__, __LINE__) &&
        cuda_ok(cudaEventRecord(s.ready, s.stream), __FILE__, __LINE__);
      if (!ok) {
        cusparseDestroyDnMat(b); cusparseDestroyDnMat(q); cusparseDestroySpMat(mat);
        return false;
      }
      s.mat_desc = mat; s.q_desc = q; s.b_desc = b;
      s.dynamic_task = -1; s.dynamic_partition = task.partition; s.dynamic_state = 1;
      return true;
    };
    auto enqueue_free_slots = [&]() -> bool {
      for (int i = 0; i < pool_slots; ++i) {
        if (slots[i].dynamic_state == 2 && cudaEventQuery(slots[i].free_event) == cudaSuccess) {
          if (slots[i].mat_desc) { cusparseDestroySpMat(slots[i].mat_desc); slots[i].mat_desc = nullptr; }
          if (slots[i].q_desc) { cusparseDestroyDnMat(slots[i].q_desc); slots[i].q_desc = nullptr; }
          if (slots[i].b_desc) { cusparseDestroyDnMat(slots[i].b_desc); slots[i].b_desc = nullptr; }
          slots[i].dynamic_state = 0; ++dynamic_completed;
        }
      }
      for (int i = 0; i < pool_slots && next_dynamic < dyn_tasks.size(); ++i) {
        if (slots[i].dynamic_state == 0) {
          if (!submit_dynamic(i, dyn_tasks[next_dynamic])) return false;
          ++next_dynamic;
        }
      }
      return true;
    };
    if (!enqueue_free_slots()) failed = true;
    while (!failed && dynamic_completed < static_cast<int>(dyn_tasks.size())) {
      // First reap reduction completions and refill slots.  No host-side wait
      // is used here; an unavailable event simply yields to CUDA progress.
      if (!enqueue_free_slots()) { failed = true; break; }
      int chosen_partition = -1;
      std::vector<int> ready_slots;
      for (int i = 0; i < pool_slots; ++i) {
        if (slots[i].dynamic_state == 1 && cudaEventQuery(slots[i].ready) == cudaSuccess) {
          if (chosen_partition < 0) chosen_partition = slots[i].dynamic_partition;
          if (slots[i].dynamic_partition == chosen_partition) ready_slots.push_back(i);
        }
      }
      if (ready_slots.empty()) {
        // If all active tasks are still computing, wait on one concrete slot
        // event.  This is a slot-local dependency, not a group-boundary
        // synchronize, and prevents CPU spin while CUDA makes progress.
        int wait_slot = -1;
        for (int i = 0; i < pool_slots; ++i) if (slots[i].dynamic_state == 1) { wait_slot = i; break; }
        if (wait_slot < 0) for (int i = 0; i < pool_slots; ++i) if (slots[i].dynamic_state == 2) { wait_slot = i; break; }
        if (wait_slot >= 0) {
          if (slots[wait_slot].dynamic_state == 1) CUDA_MAIN(cudaEventSynchronize(slots[wait_slot].ready));
          else CUDA_MAIN(cudaEventSynchronize(slots[wait_slot].free_event));
          ++forced_sync;
        }
        ++polls;
        continue;
      }
      for (int i : ready_slots) { if (!cuda_ok(cudaStreamWaitEvent(reduce_stream, slots[i].ready, 0), __FILE__, __LINE__)) { failed = true; break; } }
      if (failed) break;
      // Use a synchronous copy for the small index list.  The vector is a
      // loop-local pageable buffer; completing this copy before enqueueing the
      // tree avoids relying on pageable-host staging lifetime.
      int* group_indices = d_indices + static_cast<size_t>(ready_slots.front()) * pool_slots;
      CUDA_MAIN(cudaMemcpy(group_indices, ready_slots.data(), sizeof(int) * ready_slots.size(), cudaMemcpyHostToDevice));
      const int active = static_cast<int>(ready_slots.size());
      const int pstart = static_cast<int>((static_cast<long long>(chosen_partition) * a.cols) / partitions);
      const int pend = static_cast<int>((static_cast<long long>(chosen_partition + 1) * a.cols) / partitions);
      const int pcols = pend - pstart; const size_t n = static_cast<size_t>(pcols) * l;
      for (int stride = 1; stride < active; stride *= 2) {
        const int pairs = (active + 2 * stride - 1) / (2 * stride);
        dim3 block(8, 32); dim3 grid((pairs + block.x - 1) / block.x, 1);
        tree_level_indexed<<<grid, block, 0, reduce_stream>>>(d_pool, leaf_elements, group_indices, stride, active, n);
        if (!cuda_ok(cudaGetLastError(), __FILE__, __LINE__)) { failed = true; break; }
      }
      if (failed) break;
      const int threads = 256, blocks = static_cast<int>((n + threads - 1) / threads);
      add_to_b<<<blocks, threads, 0, reduce_stream>>>(d_pool + static_cast<size_t>(ready_slots.front()) * leaf_elements,
                                                       d_b, a.cols, pstart, pcols, l);
      if (!cuda_ok(cudaGetLastError(), __FILE__, __LINE__)) { failed = true; break; }
      for (int i : ready_slots) { if (!cuda_ok(cudaEventRecord(slots[i].free_event, reduce_stream), __FILE__, __LINE__)) { failed = true; break; } slots[i].dynamic_state = 2; }
      ++tree_groups; completed += active; ++polls;
    }
    if (!failed) CUDA_MAIN(cudaStreamSynchronize(reduce_stream));
    if (!failed) { for (Slot& s : slots) if (s.dynamic_state != 0) s.dynamic_state = 0; completed = static_cast<int>(dyn_tasks.size()); }
  }
  if (!failed) CUDA_MAIN(cudaMemcpy(h_b.data(), d_b, sizeof(float) * b_elements, cudaMemcpyDeviceToHost));
  if (!failed && integrated) {
    cusolverDnHandle_t svd_solver = nullptr; float* d_sigma = nullptr; float* d_svd_work = nullptr; int* d_svd_info = nullptr; int svd_work = 0;
    SOLVER_MAIN(cusolverDnCreate(&svd_solver)); SOLVER_MAIN(cusolverDnSgesvd_bufferSize(svd_solver, a.cols, l, &svd_work)); CUDA_MAIN(cudaMalloc(&d_sigma, sizeof(float) * l)); CUDA_MAIN(cudaMalloc(&d_svd_work, sizeof(float) * svd_work)); CUDA_MAIN(cudaMalloc(&d_svd_info, sizeof(int)));
    const auto svd_start = std::chrono::steady_clock::now(); SOLVER_MAIN(cusolverDnSgesvd(svd_solver, 'N', 'N', a.cols, l, d_b, a.cols, d_sigma, nullptr, a.cols, nullptr, l, d_svd_work, svd_work, nullptr, d_svd_info)); CUDA_MAIN(cudaDeviceSynchronize()); svd_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - svd_start).count(); CUDA_MAIN(cudaMemcpy(&svd_info, d_svd_info, sizeof(int), cudaMemcpyDeviceToHost));
    integrated_device_bytes += sizeof(float) * l + sizeof(float) * svd_work + sizeof(int); cusolverDnDestroy(svd_solver); cudaFree(d_svd_info); cudaFree(d_svd_work); cudaFree(d_sigma);
  }
  const double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall_start).count();
  std::vector<float> ref(b_elements, 0.0F);
  for (int r = 0; r < a.rows; ++r) for (int j = a.offsets[r]; j < a.offsets[r + 1]; ++j)
    for (int c = 0; c < l; ++c) ref[a.columns[j] + static_cast<size_t>(c) * a.cols] += a.values[j] * h_q[r + static_cast<size_t>(c) * a.rows];
  double d2 = 0.0, r2 = 0.0; for (size_t i = 0; i < ref.size(); ++i) { const double d = static_cast<double>(h_b[i]) - ref[i]; d2 += d * d; r2 += static_cast<double>(ref[i]) * ref[i]; }
  const double error = std::sqrt(d2 / std::max(r2, 1e-30));
  size_t free_after = 0, unused_total = 0; CUDA_MAIN(cudaMemGetInfo(&free_after, &unused_total));
  std::ofstream out(output);
  out << "input,rows,cols,nnz,rank,oversampling,block_rows,partitions,pool_slots,blocks,tree_groups,completed,polls,forced_sync,integrated_pipeline,partitioned_b,memory_pool,device_tree_reduction,heterogeneous_scheduler,cost_sorted_tasks,fifo_load_imbalance,lpt_load_imbalance,qr_ms,tree_atq_ms,svd_ms,integrated_total_ms,qr_info,svd_info,q_orthogonality_error,partial_pool_mib,b_mib,pinned_host_mib,tracked_device_mib,free_before_mib,free_after_mib,relative_error\n";
  out << input << ',' << a.rows << ',' << a.cols << ',' << a.values.size() << ',' << rank << ',' << oversampling << ',' << block_rows << ',' << partitions << ',' << pool_slots << ',' << tasks << ',' << tree_groups << ',' << completed << ',' << polls << ',' << forced_sync << ',' << (integrated ? "true" : "false") << ",true,true,true," << (dynamic_mode ? "completion_event_dynamic_queue" : "static_lpt_worker_queue") << ',' << cost_sorted_tasks << ',' << fifo_load_imbalance << ',' << lpt_load_imbalance << ',' << qr_ms << ',' << wall_ms << ',' << svd_ms << ',' << (qr_ms + wall_ms + svd_ms) << ',' << qr_info << ',' << svd_info << ',' << q_orthogonality << ',' << (sizeof(float) * leaf_elements * pool_slots / (1024.0 * 1024.0)) << ',' << (sizeof(float) * b_elements / (1024.0 * 1024.0)) << ',' << (pinned_total / (1024.0 * 1024.0)) << ',' << ((tracked_device + integrated_device_bytes) / (1024.0 * 1024.0)) << ',' << (free_before / (1024.0 * 1024.0)) << ',' << (free_after / (1024.0 * 1024.0)) << ',' << error << '\n';
  std::printf("Tree A^TQ: blocks=%d partitions=%d pool_slots=%d groups=%d completed=%d polls=%d forced_sync=%d integrated=%d QR=%.3f ms tree=%.3f ms SVD=%.3f ms total=%.3f ms pool=%.2f MiB B=%.2f MiB device=%.2f MiB rel_error=%.3e info=%d/%d Qorth=%.3e\n", tasks, partitions, pool_slots, tree_groups, completed, polls, forced_sync, integrated ? 1 : 0, qr_ms, wall_ms, svd_ms, qr_ms + wall_ms + svd_ms, sizeof(float) * leaf_elements * pool_slots / (1024.0 * 1024.0), sizeof(float) * b_elements / (1024.0 * 1024.0), (tracked_device + integrated_device_bytes) / (1024.0 * 1024.0), error, qr_info, svd_info, q_orthogonality);
  for (Slot& s : slots) destroy_slot(&s); cudaFree(d_indices); cudaFree(d_pool); cudaFree(d_b); cudaStreamDestroy(reduce_stream);
  return failed ? 1 : (error > 1e-5 ? 2 : 0);
}
