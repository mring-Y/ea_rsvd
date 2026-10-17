#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

struct CsrMatrix {
  int rows = 0, cols = 0;
  std::vector<int> offsets, columns;
  std::vector<float> values;
};

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
  out->offsets.assign(out->rows + 1, 0); out->columns.reserve(nnz); out->values.reserve(nnz);
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
      int c = col(gen); if (std::find(chosen.begin(), chosen.end(), c) == chosen.end()) chosen.push_back(c);
    }
    std::sort(chosen.begin(), chosen.end());
    for (int c : chosen) { a.columns.push_back(c); a.values.push_back(val(gen)); }
    a.offsets[r + 1] = static_cast<int>(a.columns.size());
  }
  return a;
}

static bool cu_ok(cudaError_t e, const char* file, int line) {
  if (e == cudaSuccess) return true; std::fprintf(stderr, "CUDA %s:%d: %s\n", file, line, cudaGetErrorString(e)); return false;
}
static bool event_query(cudaEvent_t event, bool* ready, const char* file, int line) {
  const cudaError_t e = cudaEventQuery(event);
  if (e == cudaSuccess) { *ready = true; return true; }
  if (e == cudaErrorNotReady) { *ready = false; return true; }
  std::fprintf(stderr, "CUDA event query %s:%d: %s\n", file, line, cudaGetErrorString(e));
  return false;
}
static bool sp_ok(cusparseStatus_t e, const char* file, int line) {
  if (e == CUSPARSE_STATUS_SUCCESS) return true; std::fprintf(stderr, "cuSPARSE %s:%d: %d\n", file, line, static_cast<int>(e)); return false;
}
static bool sol_ok(cusolverStatus_t e, const char* file, int line) {
  if (e == CUSOLVER_STATUS_SUCCESS) return true; std::fprintf(stderr, "cuSOLVER %s:%d: %d\n", file, line, static_cast<int>(e)); return false;
}
#define CU(x) do { if (!cu_ok((x), __FILE__, __LINE__)) return false; } while (0)
#define SP(x) do { if (!sp_ok((x), __FILE__, __LINE__)) return false; } while (0)
#define SO(x) do { if (!sol_ok((x), __FILE__, __LINE__)) return false; } while (0)

__global__ static void fill_omega(float* dst, int n, int l, unsigned seed) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = static_cast<size_t>(n) * l; if (i >= total) return;
  unsigned x = static_cast<unsigned>(i) ^ seed;
  x = 1664525u * x + 1013904223u; const float u = (static_cast<float>(x & 0x00ffffffu) + 1.0F) / 16777217.0F;
  x = 1664525u * x + 1013904223u; const float v = (static_cast<float>(x & 0x00ffffffu) + 1.0F) / 16777217.0F;
  dst[i] = sqrtf(-2.0F * logf(u)) * cosf(6.28318530718F * v);
}

__global__ static void tree_level_indexed(float* pool, float* compensation, size_t stride_bytes, const int* indices, int stride, int active, int pool_slots, size_t n) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x; const int base = pair * 2 * stride;
  if (base >= active || base + stride >= active) return;
  const int dst_index = indices[base];
  const int src_index = indices[base + stride];
  // Host-side validation catches bad scheduler state; retain a device guard
  // so stale/corrupt index data cannot form an out-of-range pool pointer.
  if (dst_index < 0 || dst_index >= pool_slots || src_index < 0 || src_index >= pool_slots) return;
  float* dst = pool + static_cast<size_t>(dst_index) * stride_bytes;
  const float* src = pool + static_cast<size_t>(src_index) * stride_bytes;
#ifdef KAHAN_TREE
  const size_t dst_slot = static_cast<size_t>(dst_index) * stride_bytes;
#endif
  for (size_t i = blockIdx.y * blockDim.y + threadIdx.y; i < n; i += blockDim.y * gridDim.y) {
#ifdef KAHAN_TREE
    float& sum = dst[i];
    float& corr = compensation[dst_slot + i];
    const float y = src[i] - corr;
    const float t = sum + y;
    corr = (t - sum) - y;
    sum = t;
#else
    dst[i] += src[i];
#endif
  }
}

__global__ static void add_partition(const float* src, float* b, int b_rows, int pstart, int pcols, int l) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; const size_t n = static_cast<size_t>(pcols) * l;
  if (pstart < 0 || pcols <= 0 || l <= 0 || pstart + pcols > b_rows) return;
  if (i < n) { const int local_col = static_cast<int>(i % pcols); const int c = static_cast<int>(i / pcols); b[pstart + local_col + static_cast<size_t>(c) * b_rows] += src[i]; }
}

struct Slot {
  cudaStream_t stream = nullptr; cusparseHandle_t sparse = nullptr;
  cudaEvent_t ready = nullptr, free_event = nullptr;
  cusparseSpMatDescr_t mat_desc = nullptr;
  cusparseDnMatDescr_t q_desc = nullptr, b_desc = nullptr;
  int* h_indices = nullptr;
  int* h_offsets = nullptr; int* h_columns = nullptr; float* h_values = nullptr;
  int* d_offsets = nullptr; int* d_columns = nullptr; float* d_values = nullptr; float* d_q = nullptr;
  void* d_spmm = nullptr; size_t spmm_bytes = 0; int state = 0; int partition = -1;
};

struct Lane {
  cudaStream_t qr_stream = nullptr, svd_stream = nullptr, reduce_stream = nullptr;
  cusparseHandle_t qr_sparse = nullptr; cusolverDnHandle_t qr_solver = nullptr; cusolverDnHandle_t svd_solver = nullptr;
  cusparseSpMatDescr_t matrix = nullptr; cusparseDnMatDescr_t omega = nullptr, y = nullptr;
  int* d_offsets = nullptr; int* d_columns = nullptr; float* d_values = nullptr;
  float* d_omega = nullptr; float* d_y = nullptr; float* d_b = nullptr; float* d_b_check = nullptr;
  float* d_tau = nullptr; float* d_sigma = nullptr; float* d_qr_work = nullptr; float* d_svd_work = nullptr;
  int* d_qr_info = nullptr; int* d_svd_info = nullptr;
  // cuSOLVER may require different workspace sizes for geqrf and orgqr.
  // Keep the allocation at the maximum, but pass each routine its own size.
  int qr_geqrf_work = 0, qr_orgqr_work = 0, qr_work = 0, svd_work = 0;
  void* d_qr_spmm = nullptr; size_t qr_spmm_bytes = 0;
  cudaEvent_t run_start = nullptr, qr_start = nullptr, qr_end = nullptr, tree_start = nullptr, tree_end = nullptr, svd_start = nullptr, svd_end = nullptr;
  float* d_pool = nullptr; float* d_compensation = nullptr; int* d_indices = nullptr; std::vector<Slot> slots;
};

static void destroy_lane(Lane* x, int max_rows, int max_nnz) {
  for (Slot& s : x->slots) {
    if (s.mat_desc) cusparseDestroySpMat(s.mat_desc);
    if (s.q_desc) cusparseDestroyDnMat(s.q_desc);
    if (s.b_desc) cusparseDestroyDnMat(s.b_desc);
    if (s.ready) cudaEventDestroy(s.ready); if (s.free_event) cudaEventDestroy(s.free_event); if (s.sparse) cusparseDestroy(s.sparse); if (s.stream) cudaStreamDestroy(s.stream);
    cudaFreeHost(s.h_offsets); cudaFreeHost(s.h_columns); cudaFreeHost(s.h_values);
    cudaFreeHost(s.h_indices);
    cudaFree(s.d_offsets); cudaFree(s.d_columns); cudaFree(s.d_values); cudaFree(s.d_q); cudaFree(s.d_spmm);
  }
  if (x->run_start) cudaEventDestroy(x->run_start); if (x->qr_start) cudaEventDestroy(x->qr_start); if (x->qr_end) cudaEventDestroy(x->qr_end); if (x->tree_start) cudaEventDestroy(x->tree_start); if (x->tree_end) cudaEventDestroy(x->tree_end); if (x->svd_start) cudaEventDestroy(x->svd_start); if (x->svd_end) cudaEventDestroy(x->svd_end);
  if (x->matrix) cusparseDestroySpMat(x->matrix); if (x->omega) cusparseDestroyDnMat(x->omega); if (x->y) cusparseDestroyDnMat(x->y);
  if (x->qr_sparse) cusparseDestroy(x->qr_sparse); if (x->qr_solver) cusolverDnDestroy(x->qr_solver); if (x->svd_solver) cusolverDnDestroy(x->svd_solver);
  if (x->qr_stream) cudaStreamDestroy(x->qr_stream); if (x->svd_stream) cudaStreamDestroy(x->svd_stream); if (x->reduce_stream) cudaStreamDestroy(x->reduce_stream);
  cudaFree(x->d_offsets); cudaFree(x->d_columns); cudaFree(x->d_values); cudaFree(x->d_omega); cudaFree(x->d_y); cudaFree(x->d_b); cudaFree(x->d_b_check); cudaFree(x->d_tau); cudaFree(x->d_sigma); cudaFree(x->d_qr_work); cudaFree(x->d_svd_work); cudaFree(x->d_qr_info); cudaFree(x->d_svd_info); cudaFree(x->d_qr_spmm); cudaFree(x->d_pool); cudaFree(x->d_compensation); cudaFree(x->d_indices); *x = {};
  (void)max_rows; (void)max_nnz;
}

static bool init_lane(Lane* x, const CsrMatrix& a, int l, int partitions, int slots_count, int block_rows, int max_nnz, const int* d_offsets, const int* d_columns, const float* d_values) {
  CU(cudaStreamCreateWithFlags(&x->qr_stream, cudaStreamNonBlocking)); CU(cudaStreamCreateWithFlags(&x->svd_stream, cudaStreamNonBlocking)); CU(cudaStreamCreateWithFlags(&x->reduce_stream, cudaStreamNonBlocking));
  SP(cusparseCreate(&x->qr_sparse)); SO(cusolverDnCreate(&x->qr_solver)); SO(cusolverDnCreate(&x->svd_solver)); SP(cusparseSetStream(x->qr_sparse, x->qr_stream)); SO(cusolverDnSetStream(x->qr_solver, x->qr_stream)); SO(cusolverDnSetStream(x->svd_solver, x->svd_stream));
  CU(cudaEventCreate(&x->run_start)); CU(cudaEventCreate(&x->qr_start)); CU(cudaEventCreate(&x->qr_end)); CU(cudaEventCreate(&x->tree_start)); CU(cudaEventCreate(&x->tree_end)); CU(cudaEventCreate(&x->svd_start)); CU(cudaEventCreate(&x->svd_end));
  CU(cudaMalloc(&x->d_offsets, sizeof(int) * a.offsets.size())); CU(cudaMalloc(&x->d_columns, sizeof(int) * a.columns.size())); CU(cudaMalloc(&x->d_values, sizeof(float) * a.values.size()));
  CU(cudaMemcpy(x->d_offsets, a.offsets.data(), sizeof(int) * a.offsets.size(), cudaMemcpyHostToDevice)); CU(cudaMemcpy(x->d_columns, a.columns.data(), sizeof(int) * a.columns.size(), cudaMemcpyHostToDevice)); CU(cudaMemcpy(x->d_values, a.values.data(), sizeof(float) * a.values.size(), cudaMemcpyHostToDevice));
  CU(cudaMalloc(&x->d_omega, sizeof(float) * static_cast<size_t>(a.cols) * l)); CU(cudaMalloc(&x->d_y, sizeof(float) * static_cast<size_t>(a.rows) * l)); CU(cudaMalloc(&x->d_b, sizeof(float) * static_cast<size_t>(a.cols) * l)); CU(cudaMalloc(&x->d_b_check, sizeof(float) * static_cast<size_t>(a.cols) * l)); CU(cudaMalloc(&x->d_tau, sizeof(float) * l)); CU(cudaMalloc(&x->d_qr_info, sizeof(int))); CU(cudaMalloc(&x->d_svd_info, sizeof(int))); CU(cudaMalloc(&x->d_sigma, sizeof(float) * l));
  SP(cusparseCreateCsr(&x->matrix, a.rows, a.cols, static_cast<int>(a.values.size()), const_cast<int*>(d_offsets), const_cast<int*>(d_columns), const_cast<float*>(d_values), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  SP(cusparseCreateDnMat(&x->omega, a.cols, l, a.cols, x->d_omega, CUDA_R_32F, CUSPARSE_ORDER_COL)); SP(cusparseCreateDnMat(&x->y, a.rows, l, a.rows, x->d_y, CUDA_R_32F, CUSPARSE_ORDER_COL));
  const float one = 1.0F, zero = 0.0F; SP(cusparseSpMM_bufferSize(x->qr_sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, x->matrix, x->omega, &zero, x->y, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &x->qr_spmm_bytes)); CU(cudaMalloc(&x->d_qr_spmm, x->qr_spmm_bytes));
  SO(cusolverDnSgeqrf_bufferSize(x->qr_solver, a.rows, l, x->d_y, a.rows, &x->qr_geqrf_work));
  SO(cusolverDnSorgqr_bufferSize(x->qr_solver, a.rows, l, l, x->d_y, a.rows, x->d_tau, &x->qr_orgqr_work));
  x->qr_work = std::max(x->qr_geqrf_work, x->qr_orgqr_work);
  CU(cudaMalloc(&x->d_qr_work, sizeof(float) * static_cast<size_t>(x->qr_work)));
  SO(cusolverDnSgesvd_bufferSize(x->svd_solver, a.cols, l, &x->svd_work)); CU(cudaMalloc(&x->d_svd_work, sizeof(float) * x->svd_work));
  const int max_rows = std::min(block_rows, a.rows); const int max_part_cols = (a.cols + partitions - 1) / partitions; const size_t leaf = static_cast<size_t>(max_part_cols) * l;
  CU(cudaMalloc(&x->d_pool, sizeof(float) * leaf * slots_count));
#ifdef KAHAN_TREE
  CU(cudaMalloc(&x->d_compensation, sizeof(float) * leaf * slots_count));
#endif
  CU(cudaMalloc(&x->d_indices, sizeof(int) * slots_count * slots_count)); x->slots.resize(slots_count);
  for (Slot& s : x->slots) {
    CU(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking)); SP(cusparseCreate(&s.sparse)); SP(cusparseSetStream(s.sparse, s.stream)); CU(cudaEventCreateWithFlags(&s.ready, cudaEventDisableTiming)); CU(cudaEventCreateWithFlags(&s.free_event, cudaEventDisableTiming));
    CU(cudaMallocHost(&s.h_offsets, sizeof(int) * (max_rows + 1))); CU(cudaMallocHost(&s.h_columns, sizeof(int) * max_nnz)); CU(cudaMallocHost(&s.h_values, sizeof(float) * max_nnz));
    CU(cudaMallocHost(&s.h_indices, sizeof(int) * slots_count));
    CU(cudaMalloc(&s.d_offsets, sizeof(int) * (max_rows + 1))); CU(cudaMalloc(&s.d_columns, sizeof(int) * max_nnz)); CU(cudaMalloc(&s.d_values, sizeof(float) * max_nnz)); CU(cudaMalloc(&s.d_q, sizeof(float) * max_rows * l));
  }
  return true;
}

struct Task { int block = 0, partition = 0, cost = 0; };

static bool submit_task(Lane* x, Slot& s, const CsrMatrix& a, int l, int block_rows, int partitions, int task_id, const std::vector<Task>& tasks) {
  const Task& task = tasks[task_id]; const int pstart = static_cast<int>((static_cast<long long>(task.partition) * a.cols) / partitions); const int pend = static_cast<int>((static_cast<long long>(task.partition + 1) * a.cols) / partitions); const int pcols = pend - pstart; const int first = task.block * block_rows; const int rows = std::min(block_rows, a.rows - first);
  int local_nnz = 0; s.h_offsets[0] = 0;
  for (int r = 0; r < rows; ++r) { for (int j = a.offsets[first + r]; j < a.offsets[first + r + 1]; ++j) if (a.columns[j] >= pstart && a.columns[j] < pend) { s.h_columns[local_nnz] = a.columns[j] - pstart; s.h_values[local_nnz] = a.values[j]; ++local_nnz; } s.h_offsets[r + 1] = local_nnz; }
  // Empty row-block/partition tiles are mathematically zero.  Avoid passing
  // an nnz=0 CSR descriptor to cuSPARSE, which is rejected by some CUDA
  // toolkit versions, while preserving the same ready-event dependency.
  const int slot_index = static_cast<int>(&s - x->slots.data());
  const size_t leaf = ((static_cast<size_t>(a.cols) + partitions - 1) / partitions) * l;
  if (local_nnz == 0) {
    bool ok = cu_ok(cudaStreamWaitEvent(s.stream, x->qr_end, 0), __FILE__, __LINE__) &&
              cu_ok(cudaMemsetAsync(x->d_pool + static_cast<size_t>(slot_index) * leaf,
                                    0, sizeof(float) * leaf, s.stream), __FILE__, __LINE__);
#ifdef KAHAN_TREE
    const size_t slot_base = static_cast<size_t>(slot_index) * leaf;
    ok = ok && cu_ok(cudaMemsetAsync(x->d_compensation + slot_base, 0,
                                     sizeof(float) * leaf, s.stream), __FILE__, __LINE__);
#endif
    ok = ok && cu_ok(cudaEventRecord(s.ready, s.stream), __FILE__, __LINE__);
    if (!ok) return false;
    s.partition = task.partition; s.state = 1;
    return true;
  }
  cusparseSpMatDescr_t mat = nullptr; cusparseDnMatDescr_t q = nullptr, b = nullptr; const float one = 1.0F, zero = 0.0F; size_t bytes = 0;
  if (!sp_ok(cusparseCreateCsr(&mat, rows, pcols, local_nnz, s.d_offsets, s.d_columns, s.d_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F), __FILE__, __LINE__) || !sp_ok(cusparseCreateDnMat(&q, rows, l, rows, s.d_q, CUDA_R_32F, CUSPARSE_ORDER_COL), __FILE__, __LINE__) || !sp_ok(cusparseCreateDnMat(&b, pcols, l, pcols, x->d_pool + static_cast<size_t>(&s - x->slots.data()) * ((static_cast<size_t>(a.cols) + partitions - 1) / partitions) * l, CUDA_R_32F, CUSPARSE_ORDER_COL), __FILE__, __LINE__)) return false;
  if (!sp_ok(cusparseSpMM_bufferSize(s.sparse, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat, q, &zero, b, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bytes), __FILE__, __LINE__)) return false;
  if (bytes > s.spmm_bytes) { cudaFree(s.d_spmm); if (!cu_ok(cudaMalloc(&s.d_spmm, bytes), __FILE__, __LINE__)) return false; s.spmm_bytes = bytes; }
  bool ok = cu_ok(cudaStreamWaitEvent(s.stream, x->qr_end, 0), __FILE__, __LINE__) && cu_ok(cudaMemcpyAsync(s.d_offsets, s.h_offsets, sizeof(int) * (rows + 1), cudaMemcpyHostToDevice, s.stream), __FILE__, __LINE__) && cu_ok(cudaMemcpyAsync(s.d_columns, s.h_columns, sizeof(int) * local_nnz, cudaMemcpyHostToDevice, s.stream), __FILE__, __LINE__) && cu_ok(cudaMemcpyAsync(s.d_values, s.h_values, sizeof(float) * local_nnz, cudaMemcpyHostToDevice, s.stream), __FILE__, __LINE__);
#ifdef KAHAN_TREE
  const size_t slot_base = static_cast<size_t>(&s - x->slots.data()) * leaf;
  ok = ok && cu_ok(cudaMemsetAsync(x->d_compensation + slot_base, 0, sizeof(float) * leaf, s.stream), __FILE__, __LINE__);
#endif
  for (int c = 0; ok && c < l; ++c) ok = cu_ok(cudaMemcpyAsync(s.d_q + static_cast<size_t>(c) * rows, x->d_y + first + static_cast<size_t>(c) * a.rows, sizeof(float) * rows, cudaMemcpyDeviceToDevice, s.stream), __FILE__, __LINE__);
  ok = ok && sp_ok(cusparseSpMM(s.sparse, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat, q, &zero, b, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, s.d_spmm), __FILE__, __LINE__) && cu_ok(cudaEventRecord(s.ready, s.stream), __FILE__, __LINE__);
  if (!ok) {
    if (b) cusparseDestroyDnMat(b); if (q) cusparseDestroyDnMat(q); if (mat) cusparseDestroySpMat(mat);
    return false;
  }
  s.mat_desc = mat; s.q_desc = q; s.b_desc = b;
  s.partition = task.partition; s.state = 1; (void)slot_index; (void)leaf; return true;
}

static bool dynamic_tree(Lane* x, const CsrMatrix& a, int l, int block_rows, int partitions, int pool_slots, double* polls_out, double* waits_out) {
  const int tasks_count = (a.rows + block_rows - 1) / block_rows; std::vector<Task> tasks; tasks.reserve(static_cast<size_t>(tasks_count) * partitions);
  for (int p = 0; p < partitions; ++p) for (int t = 0; t < tasks_count; ++t) { const int first = t * block_rows, rows = std::min(block_rows, a.rows - first); int cost = 0; const int pstart = static_cast<int>((static_cast<long long>(p) * a.cols) / partitions), pend = static_cast<int>((static_cast<long long>(p + 1) * a.cols) / partitions); for (int r = first; r < first + rows; ++r) for (int j = a.offsets[r]; j < a.offsets[r + 1]; ++j) cost += (a.columns[j] >= pstart && a.columns[j] < pend); tasks.push_back({t, p, cost}); }
  std::stable_sort(tasks.begin(), tasks.end(), [](const Task& x, const Task& y) { return x.cost > y.cost; });
  const size_t leaf = ((static_cast<size_t>(a.cols) + partitions - 1) / partitions) * l; CU(cudaMemsetAsync(x->d_b, 0, sizeof(float) * static_cast<size_t>(a.cols) * l, x->reduce_stream)); CU(cudaEventRecord(x->tree_start, x->reduce_stream));
  size_t next = 0; int completed = 0; int polls = 0, waits = 0;
  // A lane is reused for warm-up, sequential, and parallel measurements.
  // The preceding reduction has already recorded each slot's free event, but
  // its cuSPARSE descriptors still belong to that completed submission.  Wait
  // for the event and release those descriptors before making the slot free;
  // resetting state alone leaks descriptors across pipeline rounds.
  for (Slot& s : x->slots) {
    if (s.state == 2) {
      if (!cu_ok(cudaEventSynchronize(s.free_event), __FILE__, __LINE__)) return false;
      if (s.mat_desc) { cusparseDestroySpMat(s.mat_desc); s.mat_desc = nullptr; }
      if (s.q_desc) { cusparseDestroyDnMat(s.q_desc); s.q_desc = nullptr; }
      if (s.b_desc) { cusparseDestroyDnMat(s.b_desc); s.b_desc = nullptr; }
    }
    s.state = 0;
  }
  auto fill = [&]() -> bool {
    for (int i = 0; i < pool_slots && next < tasks.size(); ++i) {
      Slot& s = x->slots[i];
      bool free_ready = false;
      if (s.state == 2 && !event_query(s.free_event, &free_ready, __FILE__, __LINE__)) return false;
      if (s.state == 2 && free_ready) {
        if (s.mat_desc) { cusparseDestroySpMat(s.mat_desc); s.mat_desc = nullptr; }
        if (s.q_desc) { cusparseDestroyDnMat(s.q_desc); s.q_desc = nullptr; }
        if (s.b_desc) { cusparseDestroyDnMat(s.b_desc); s.b_desc = nullptr; }
        s.state = 0;
      }
      if (s.state == 0) { if (!submit_task(x, s, a, l, block_rows, partitions, static_cast<int>(next), tasks)) return false; ++next; }
    }
    return true;
  };
  if (!fill()) return false;
  while (completed < static_cast<int>(tasks.size())) {
    if (!fill()) return false; std::vector<int> ready; int p = -1;
    for (int i = 0; i < pool_slots; ++i) {
      bool ready_now = false;
      if (x->slots[i].state == 1 && !event_query(x->slots[i].ready, &ready_now, __FILE__, __LINE__)) return false;
      if (x->slots[i].state == 1 && ready_now) { if (p < 0) p = x->slots[i].partition; if (x->slots[i].partition == p) ready.push_back(i); }
    }
    if (ready.empty()) { int wait_slot = -1; for (int i = 0; i < pool_slots; ++i) if (x->slots[i].state == 1) { wait_slot = i; break; } if (wait_slot < 0) for (int i = 0; i < pool_slots; ++i) if (x->slots[i].state == 2) { wait_slot = i; break; } if (wait_slot >= 0) { if (x->slots[wait_slot].state == 1) CU(cudaEventSynchronize(x->slots[wait_slot].ready)); else CU(cudaEventSynchronize(x->slots[wait_slot].free_event)); ++waits; } ++polls; continue; }
    for (int i : ready) CU(cudaStreamWaitEvent(x->reduce_stream, x->slots[i].ready, 0));
    const int active = static_cast<int>(ready.size());
    const int pstart = static_cast<int>((static_cast<long long>(p) * a.cols) / partitions);
    const int pcols = static_cast<int>((static_cast<long long>(p + 1) * a.cols) / partitions) - pstart;
    const size_t n = static_cast<size_t>(pcols) * l;
    if (active <= 0 || active > pool_slots || p < 0 || p >= partitions || pstart < 0 ||
        pcols <= 0 || pstart + pcols > a.cols ||
        ready.front() < 0 || ready.front() >= pool_slots) {
      std::fprintf(stderr, "dynamic_tree invalid group: partition=%d active=%d pool_slots=%d pcols=%d first=%d\n",
                   p, active, pool_slots, pcols, ready.empty() ? -1 : ready.front());
      return false;
    }
    for (int slot : ready) {
      if (slot < 0 || slot >= pool_slots) {
        std::fprintf(stderr, "dynamic_tree invalid slot: partition=%d slot=%d pool_slots=%d\n", p, slot, pool_slots);
        return false;
      }
    }
    Slot& index_slot = x->slots[ready.front()];
    std::copy(ready.begin(), ready.end(), index_slot.h_indices);
    int* group_indices = x->d_indices + static_cast<size_t>(ready.front()) * pool_slots;
    // The list is tiny.  A synchronous copy makes the host buffer lifetime
    // explicit even when a slot is revisited by the scheduler.
    CU(cudaMemcpy(group_indices, index_slot.h_indices, sizeof(int) * ready.size(), cudaMemcpyHostToDevice));
    for (int stride = 1; stride < active; stride *= 2) { const int pairs = (active + 2 * stride - 1) / (2 * stride); dim3 block(8, 32), grid((pairs + block.x - 1) / block.x, 1); tree_level_indexed<<<grid, block, 0, x->reduce_stream>>>(x->d_pool, x->d_compensation, leaf, group_indices, stride, active, pool_slots, n); if (!cu_ok(cudaGetLastError(), __FILE__, __LINE__)) return false; }
    const int threads = 256, blocks = static_cast<int>((n + threads - 1) / threads); const size_t first_leaf = static_cast<size_t>(ready.front()) * leaf; add_partition<<<blocks, threads, 0, x->reduce_stream>>>(x->d_pool + first_leaf, x->d_b, a.cols, pstart, pcols, l); if (!cu_ok(cudaGetLastError(), __FILE__, __LINE__)) return false;
    for (int i : ready) { CU(cudaEventRecord(x->slots[i].free_event, x->reduce_stream)); x->slots[i].state = 2; } completed += active; ++polls;
  }
  CU(cudaStreamSynchronize(x->reduce_stream)); CU(cudaEventRecord(x->tree_end, x->reduce_stream)); CU(cudaStreamWaitEvent(x->svd_stream, x->tree_end, 0)); CU(cudaEventRecord(x->svd_start, x->svd_stream)); CU(cudaMemcpyAsync(x->d_b_check, x->d_b, sizeof(float) * static_cast<size_t>(a.cols) * l, cudaMemcpyDeviceToDevice, x->svd_stream)); SO(cusolverDnSgesvd(x->svd_solver, 'N', 'N', a.cols, l, x->d_b, a.cols, x->d_sigma, nullptr, a.cols, nullptr, l, x->d_svd_work, x->svd_work, nullptr, x->d_svd_info)); CU(cudaEventRecord(x->svd_end, x->svd_stream)); CU(cudaStreamSynchronize(x->svd_stream)); *polls_out = polls; *waits_out = waits; return true;
}

static bool run_pipeline(Lane* x, const CsrMatrix& a, int l, int block_rows, int partitions, int pool_slots, unsigned seed, double* polls, double* waits) {
  CU(cudaEventRecord(x->run_start, x->qr_stream)); const int threads = 256, blocks = static_cast<int>((static_cast<size_t>(a.cols) * l + threads - 1) / threads); fill_omega<<<blocks, threads, 0, x->qr_stream>>>(x->d_omega, a.cols, l, seed); if (!cu_ok(cudaGetLastError(), __FILE__, __LINE__)) return false; CU(cudaEventRecord(x->qr_start, x->qr_stream)); const float one = 1.0F, zero = 0.0F; SP(cusparseSpMM(x->qr_sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, x->matrix, x->omega, &zero, x->y, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, x->d_qr_spmm)); SO(cusolverDnSgeqrf(x->qr_solver, a.rows, l, x->d_y, a.rows, x->d_tau, x->d_qr_work, x->qr_geqrf_work, x->d_qr_info)); SO(cusolverDnSorgqr(x->qr_solver, a.rows, l, l, x->d_y, a.rows, x->d_tau, x->d_qr_work, x->qr_orgqr_work, x->d_qr_info)); CU(cudaEventRecord(x->qr_end, x->qr_stream)); return dynamic_tree(x, a, l, block_rows, partitions, pool_slots, polls, waits);
}

static bool event_ms(cudaEvent_t a, cudaEvent_t b, double* out) { float v = 0.0F; if (!cu_ok(cudaEventElapsedTime(&v, a, b), __FILE__, __LINE__)) return false; *out = v; return true; }
static double overlap(double a0, double a1, double b0, double b1) { return std::max(0.0, std::min(a1, b1) - std::max(a0, b0)); }
static double relative_error(const CsrMatrix& a, const std::vector<float>& q, const std::vector<float>& b, int l) { std::vector<float> ref(static_cast<size_t>(a.cols) * l, 0.0F); for (int r = 0; r < a.rows; ++r) for (int j = a.offsets[r]; j < a.offsets[r + 1]; ++j) for (int c = 0; c < l; ++c) ref[a.columns[j] + static_cast<size_t>(c) * a.cols] += a.values[j] * q[r + static_cast<size_t>(c) * a.rows]; double d2 = 0.0, r2 = 0.0; for (size_t i = 0; i < ref.size(); ++i) { const double d = static_cast<double>(b[i]) - ref[i]; d2 += d * d; r2 += static_cast<double>(ref[i]) * ref[i]; } return std::sqrt(d2 / std::max(r2, 1e-30)); }
static double orth_error(const std::vector<float>& q, int rows, int l) { double worst = 0.0; for (int c1 = 0; c1 < l; ++c1) for (int c2 = 0; c2 < l; ++c2) { double dot = 0.0; for (int r = 0; r < rows; ++r) dot += static_cast<double>(q[r + static_cast<size_t>(c1) * rows]) * q[r + static_cast<size_t>(c2) * rows]; worst = std::max(worst, std::fabs(dot - (c1 == c2 ? 1.0 : 0.0))); } return worst; }
static double frobenius_sq(const CsrMatrix& a) { double s = 0.0; for (float v : a.values) { const double d = static_cast<double>(v); s += d * d; } return s; }

int main(int argc, char** argv) {
  const std::string input = argc >= 2 ? argv[1] : "synthetic"; const std::string output = argc >= 3 ? argv[2] : "dual_lane_dynamic_overlap.csv"; const int rank = argc >= 4 ? std::atoi(argv[3]) : 16; const int block_rows = argc >= 5 ? std::atoi(argv[4]) : 257; const int partitions = argc >= 6 ? std::atoi(argv[5]) : 4; const int pool_slots = argc >= 7 ? std::atoi(argv[6]) : 4; constexpr int oversampling = 8; const int l = rank + oversampling;
  CsrMatrix a; if (input == "synthetic") a = synthetic(2048, 1536, 8); else if (!read_mtx(input, &a)) { std::fprintf(stderr, "Cannot read Matrix Market input: %s\n", input.c_str()); return 65; } if (rank <= 0 || block_rows <= 0 || partitions <= 0 || partitions > a.cols || pool_slots <= 0 || l > std::min(a.rows, a.cols)) { std::fprintf(stderr, "Invalid dimensions: rows=%d cols=%d rank=%d block_rows=%d partitions=%d pool_slots=%d\n", a.rows, a.cols, rank, block_rows, partitions, pool_slots); return 65; }
  if (!cu_ok(cudaSetDevice(0), __FILE__, __LINE__)) return 1;
  int max_nnz = 0; for (int t = 0; t < (a.rows + block_rows - 1) / block_rows; ++t) { const int first = t * block_rows; const int rows = std::min(block_rows, a.rows - first); max_nnz = std::max(max_nnz, a.offsets[first + rows] - a.offsets[first]); }
  int *d_offsets = nullptr, *d_columns = nullptr; float* d_values = nullptr; CU(cudaMalloc(&d_offsets, sizeof(int) * a.offsets.size())); CU(cudaMalloc(&d_columns, sizeof(int) * a.columns.size())); CU(cudaMalloc(&d_values, sizeof(float) * a.values.size())); CU(cudaMemcpy(d_offsets, a.offsets.data(), sizeof(int) * a.offsets.size(), cudaMemcpyHostToDevice)); CU(cudaMemcpy(d_columns, a.columns.data(), sizeof(int) * a.columns.size(), cudaMemcpyHostToDevice)); CU(cudaMemcpy(d_values, a.values.data(), sizeof(float) * a.values.size(), cudaMemcpyHostToDevice));
  Lane lane[2]; for (int i = 0; i < 2; ++i) if (!init_lane(&lane[i], a, l, partitions, pool_slots, block_rows, max_nnz, d_offsets, d_columns, d_values)) { std::fprintf(stderr, "phase=init lane=%d failed\n", i); return 1; }
  double warm_p = 0.0, warm_w = 0.0; for (int i = 0; i < 2; ++i) if (!run_pipeline(&lane[i], a, l, block_rows, partitions, pool_slots, 0x1000u + i, &warm_p, &warm_w)) { std::fprintf(stderr, "phase=warmup lane=%d failed\n", i); return 1; }
  auto seq_start = std::chrono::steady_clock::now(); double seq_p = 0.0, seq_w = 0.0; if (!run_pipeline(&lane[0], a, l, block_rows, partitions, pool_slots, 0x20260928u, &seq_p, &seq_w)) { std::fprintf(stderr, "phase=sequential lane=0 failed\n"); return 1; } if (!run_pipeline(&lane[1], a, l, block_rows, partitions, pool_slots, 0x20260929u, &seq_p, &seq_w)) { std::fprintf(stderr, "phase=sequential lane=1 failed\n"); return 1; } CU(cudaDeviceSynchronize()); const double sequential_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - seq_start).count();
  auto par_start = std::chrono::steady_clock::now(); bool ok0 = false, ok1 = false; double p0 = 0.0, p1 = 0.0, w0 = 0.0, w1 = 0.0; std::thread t0([&] { if (cudaSetDevice(0) == cudaSuccess) ok0 = run_pipeline(&lane[0], a, l, block_rows, partitions, pool_slots, 0xA510u, &p0, &w0); }); std::thread t1([&] { if (cudaSetDevice(0) == cudaSuccess) ok1 = run_pipeline(&lane[1], a, l, block_rows, partitions, pool_slots, 0xB620u, &p1, &w1); }); t0.join(); t1.join(); CU(cudaDeviceSynchronize()); const double parallel_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - par_start).count(); if (!ok0 || !ok1) { std::fprintf(stderr, "phase=parallel lane0=%s lane1=%s\n", ok0 ? "ok" : "failed", ok1 ? "ok" : "failed"); return 1; }
  double qr0 = 0.0, tree0 = 0.0, svd0 = 0.0, qr1 = 0.0, tree1 = 0.0, svd1 = 0.0;
  double qrs0 = 0.0, qre0 = 0.0, qrs1 = 0.0, qre1 = 0.0, trs0 = 0.0, tre0 = 0.0, trs1 = 0.0, tre1 = 0.0, ss0 = 0.0, se0 = 0.0, ss1 = 0.0, se1 = 0.0;
  event_ms(lane[0].qr_start, lane[0].qr_end, &qr0); event_ms(lane[0].tree_start, lane[0].tree_end, &tree0); event_ms(lane[0].svd_start, lane[0].svd_end, &svd0); event_ms(lane[1].qr_start, lane[1].qr_end, &qr1); event_ms(lane[1].tree_start, lane[1].tree_end, &tree1); event_ms(lane[1].svd_start, lane[1].svd_end, &svd1);
  event_ms(lane[0].run_start, lane[0].qr_start, &qrs0); event_ms(lane[0].run_start, lane[0].qr_end, &qre0); event_ms(lane[1].run_start, lane[1].qr_start, &qrs1); event_ms(lane[1].run_start, lane[1].qr_end, &qre1);
  event_ms(lane[0].run_start, lane[0].tree_start, &trs0); event_ms(lane[0].run_start, lane[0].tree_end, &tre0); event_ms(lane[1].run_start, lane[1].tree_start, &trs1); event_ms(lane[1].run_start, lane[1].tree_end, &tre1);
  event_ms(lane[0].run_start, lane[0].svd_start, &ss0); event_ms(lane[0].run_start, lane[0].svd_end, &se0); event_ms(lane[1].run_start, lane[1].svd_start, &ss1); event_ms(lane[1].run_start, lane[1].svd_end, &se1);
  std::vector<float> hq(static_cast<size_t>(a.rows) * l), hb(static_cast<size_t>(a.cols) * l), hsigma(l); CU(cudaMemcpy(hq.data(), lane[0].d_y, sizeof(float) * hq.size(), cudaMemcpyDeviceToHost)); CU(cudaMemcpy(hb.data(), lane[0].d_b_check, sizeof(float) * hb.size(), cudaMemcpyDeviceToHost)); CU(cudaMemcpy(hsigma.data(), lane[0].d_sigma, sizeof(float) * hsigma.size(), cudaMemcpyDeviceToHost)); const double qerr = orth_error(hq, a.rows, l); const double berr = relative_error(a, hq, hb, l); const double anorm2 = frobenius_sq(a); double bnorm2 = 0.0; for (float v : hb) { const double d = static_cast<double>(v); bnorm2 += d * d; } double sigma_rank2 = 0.0; for (int i = 0; i < rank && i < l; ++i) { const double d = static_cast<double>(hsigma[i]); sigma_rank2 += d * d; } const double rho_l = std::sqrt(std::max(0.0, anorm2 - bnorm2) / std::max(anorm2, 1e-30)); const double rho_k = std::sqrt(std::max(0.0, anorm2 - sigma_rank2) / std::max(anorm2, 1e-30)); int info0 = 0, info1 = 0; CU(cudaMemcpy(&info0, lane[0].d_svd_info, sizeof(int), cudaMemcpyDeviceToHost)); CU(cudaMemcpy(&info1, lane[1].d_svd_info, sizeof(int), cudaMemcpyDeviceToHost));
  const double qr_overlap = overlap(qrs0, qre0, qrs1, qre1), tree_overlap = overlap(trs0, tre0, trs1, tre1), svd_overlap = overlap(ss0, se0, ss1, se1), pipeline_overlap = overlap(qrs0, se0, qrs1, se1);
  std::ofstream out(output); out << "input,rows,cols,nnz,rank,lanes,block_rows,partitions,pool_slots,sequential_ms,parallel_ms,speedup,qr0_ms,tree0_ms,svd0_ms,qr1_ms,tree1_ms,svd1_ms,qr_stage_overlap_ms,tree_stage_overlap_ms,svd_stage_overlap_ms,pipeline_window_overlap_ms,queue_polls0,queue_polls1,slot_waits0,slot_waits1,completed_lanes,solver_info0,solver_info1,q_orthogonality_error,atq_relative_error,rho_l,rho_k\n"; out << input << ',' << a.rows << ',' << a.cols << ',' << a.values.size() << ',' << rank << ",2," << block_rows << ',' << partitions << ',' << pool_slots << ',' << sequential_ms << ',' << parallel_ms << ',' << sequential_ms / std::max(parallel_ms, 1e-9) << ',' << qr0 << ',' << tree0 << ',' << svd0 << ',' << qr1 << ',' << tree1 << ',' << svd1 << ',' << qr_overlap << ',' << tree_overlap << ',' << svd_overlap << ',' << pipeline_overlap << ',' << p0 << ',' << p1 << ',' << w0 << ',' << w1 << ",2," << info0 << ',' << info1 << ',' << qerr << ',' << berr << ',' << rho_l << ',' << rho_k << '\n';
  std::printf("Dual-lane dynamic overlap: sequential=%.3f ms parallel=%.3f ms speedup=%.3fx QR=%.3f/%.3f tree=%.3f/%.3f SVD=%.3f/%.3f stage-overlap=%.3f/%.3f/%.3f pipeline-window=%.3f error=%.3e Qorth=%.3e info=%d/%d\n", sequential_ms, parallel_ms, sequential_ms / std::max(parallel_ms, 1e-9), qr0, qr1, tree0, tree1, svd0, svd1, qr_overlap, tree_overlap, svd_overlap, pipeline_overlap, berr, qerr, info0, info1);
  for (Lane& x : lane) destroy_lane(&x, 0, 0); cudaFree(d_values); cudaFree(d_columns); cudaFree(d_offsets); return (qerr > 1e-3 || berr > 1e-4 || info0 != 0 || info1 != 0) ? 2 : 0;
}
