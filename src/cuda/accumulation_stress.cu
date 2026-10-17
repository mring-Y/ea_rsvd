#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// Cancellation-focused A^T Q reduction benchmark.  The input is a dense-per-row
// CSR matrix whose values are exactly representable floats.  Each block contains
// large positive and negative products followed by a small residual, making the
// accumulation policy visible while keeping the sparse reduction structure fixed.

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
#define CUDA_OK(x) do { if (!cuda_ok((x), __FILE__, __LINE__)) return 1; } while (0)

enum class AccumMode { Fp32, Fp64, Kahan };

__device__ __forceinline__ void add_kahan(float x, float& sum, float& corr) {
  const float y = x - corr;
  const float t = sum + y;
  corr = (t - sum) - y;
  sum = t;
}

// One thread computes one (column, Q-column) partial for a row block.  Keeping
// this kernel deliberately simple isolates arithmetic error from sparse-library
// algorithm selection; the same block partials feed every tree variant.
template <AccumMode Mode>
__global__ void block_partials(const int* offsets, const int* columns,
                               const float* values, const float* q,
                               float* partial_f, double* partial_d,
                               int rows, int cols, int l, int block_rows,
                               int blocks) {
  const int out = blockIdx.x * blockDim.x + threadIdx.x;
  const int nout = cols * l;
  if (out >= blocks * nout) return;
  const int block = out / nout;
  const int z = out % nout;
  const int c = z % cols;
  const int k = z / cols;
  const int first = block * block_rows;
  const int last = min(rows, first + block_rows);
  if constexpr (Mode == AccumMode::Fp64) {
    double s = 0.0;
    for (int r = first; r < last; ++r)
      for (int j = offsets[r]; j < offsets[r + 1]; ++j)
        if (columns[j] == c) s += static_cast<double>(values[j]) * static_cast<double>(q[r + static_cast<size_t>(k) * rows]);
    partial_d[static_cast<size_t>(block) * nout + z] = s;
  } else {
    float s = 0.0F, corr = 0.0F;
    for (int r = first; r < last; ++r)
      for (int j = offsets[r]; j < offsets[r + 1]; ++j) {
        if (columns[j] != c) continue;
        const float x = values[j] * q[r + static_cast<size_t>(k) * rows];
        if constexpr (Mode == AccumMode::Kahan) add_kahan(x, s, corr); else s += x;
      }
    partial_f[static_cast<size_t>(block) * nout + z] = s;
  }
}

template <AccumMode Mode>
__global__ void tree_level_f(const float* src, float* dst, int active, int stride, int nout) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int pair = i / nout;
  const int z = i % nout;
  const int base = pair * 2 * stride;
  if (base >= active) return;
  const float x = src[static_cast<size_t>(base) * nout + z];
  if (base + stride >= active) {
    // Carry an odd leaf to the next level.  The fixed-stride layout keeps the
    // original leaf index, so the following level can pair it correctly.
    dst[static_cast<size_t>(base) * nout + z] = x;
    return;
  }
  const float y = src[static_cast<size_t>(base + stride) * nout + z];
  if constexpr (Mode == AccumMode::Kahan) {
    float s = x, c = 0.0F;
    add_kahan(y, s, c);
    dst[static_cast<size_t>(base) * nout + z] = s;
  } else {
    dst[static_cast<size_t>(base) * nout + z] = x + y;
  }
}

__global__ void tree_level_d(const double* src, double* dst, int active, int stride, int nout) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int pair = i / nout;
  const int z = i % nout;
  const int base = pair * 2 * stride;
  if (base >= active) return;
  if (base + stride >= active) {
    dst[static_cast<size_t>(base) * nout + z] = src[static_cast<size_t>(base) * nout + z];
    return;
  }
  dst[static_cast<size_t>(base) * nout + z] =
      src[static_cast<size_t>(base) * nout + z] + src[static_cast<size_t>(base + stride) * nout + z];
}

template <AccumMode Mode>
static int run_mode(const CsrMatrix& a, const std::vector<float>& q,
                    int block_rows, int repeats, const std::string& mode_name,
                    std::ofstream& csv, const std::vector<long double>& ref) {
  const int l = 2, nout = a.cols * l;
  const int blocks = (a.rows + block_rows - 1) / block_rows;
  const size_t partial_count = static_cast<size_t>(blocks) * nout;
  int *d_offsets = nullptr, *d_columns = nullptr;
  float *d_values = nullptr, *d_q = nullptr, *d_pf = nullptr, *d_treef = nullptr;
  double *d_pd = nullptr, *d_treed = nullptr;
  CUDA_OK(cudaMalloc(&d_offsets, sizeof(int) * a.offsets.size()));
  CUDA_OK(cudaMalloc(&d_columns, sizeof(int) * a.columns.size()));
  CUDA_OK(cudaMalloc(&d_values, sizeof(float) * a.values.size()));
  CUDA_OK(cudaMalloc(&d_q, sizeof(float) * q.size()));
  CUDA_OK(cudaMemcpy(d_offsets, a.offsets.data(), sizeof(int) * a.offsets.size(), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_columns, a.columns.data(), sizeof(int) * a.columns.size(), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_values, a.values.data(), sizeof(float) * a.values.size(), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_q, q.data(), sizeof(float) * q.size(), cudaMemcpyHostToDevice));
  if constexpr (Mode == AccumMode::Fp64) {
    CUDA_OK(cudaMalloc(&d_pd, sizeof(double) * partial_count));
    CUDA_OK(cudaMalloc(&d_treed, sizeof(double) * partial_count));
  } else {
    CUDA_OK(cudaMalloc(&d_pf, sizeof(float) * partial_count));
    CUDA_OK(cudaMalloc(&d_treef, sizeof(float) * partial_count));
  }
  std::vector<float> h_f(partial_count);
  std::vector<double> h_d(partial_count);
  double total_ms = 0.0;
  double max_abs = 0.0, max_rel = 0.0, max_sum_abs = 0.0;
  const int threads = 256;
  for (int rep = 0; rep < repeats; ++rep) {
    CUDA_OK(cudaDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    const int grid = (blocks * nout + threads - 1) / threads;
    if constexpr (Mode == AccumMode::Fp64)
      block_partials<Mode><<<grid, threads>>>(d_offsets, d_columns, d_values, d_q, nullptr, d_pd, a.rows, a.cols, l, block_rows, blocks);
    else
      block_partials<Mode><<<grid, threads>>>(d_offsets, d_columns, d_values, d_q, d_pf, nullptr, a.rows, a.cols, l, block_rows, blocks);
    CUDA_OK(cudaGetLastError());
    int levels = 0;
    for (int x = 1; x < blocks; x *= 2) ++levels;
    float* curr_f = d_pf; float* next_f = d_treef;
    double* curr_d = d_pd; double* next_d = d_treed;
    for (int stride = 1; stride < blocks; stride *= 2) {
      const int pairs = (blocks + 2 * stride - 1) / (2 * stride);
      const int tree_grid = (pairs * nout + threads - 1) / threads;
      if constexpr (Mode == AccumMode::Fp64) {
        tree_level_d<<<tree_grid, threads>>>(curr_d, next_d, blocks, stride, nout);
      } else {
        tree_level_f<Mode><<<tree_grid, threads>>>(curr_f, next_f, blocks, stride, nout);
      }
      CUDA_OK(cudaGetLastError());
      if constexpr (Mode == AccumMode::Fp64) std::swap(curr_d, next_d); else std::swap(curr_f, next_f);
    }
    CUDA_OK(cudaDeviceSynchronize());
    total_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if constexpr (Mode == AccumMode::Fp64) {
      CUDA_OK(cudaMemcpy(h_d.data(), curr_d, sizeof(double) * partial_count, cudaMemcpyDeviceToHost));
      for (int z = 0; z < nout; ++z) {
        const long double got = h_d[z];
        const long double err = std::fabs(got - ref[z]);
        max_abs = std::max(max_abs, static_cast<double>(err));
        max_rel = std::max(max_rel, static_cast<double>(err / std::max(std::fabs(ref[z]), 1e-30L)));
        max_sum_abs = std::max(max_sum_abs, std::fabs(static_cast<double>(got)));
      }
    } else {
      CUDA_OK(cudaMemcpy(h_f.data(), curr_f, sizeof(float) * partial_count, cudaMemcpyDeviceToHost));
      for (int z = 0; z < nout; ++z) {
        const long double got = h_f[z];
        const long double err = std::fabs(got - ref[z]);
        max_abs = std::max(max_abs, static_cast<double>(err));
        max_rel = std::max(max_rel, static_cast<double>(err / std::max(std::fabs(ref[z]), 1e-30L)));
        max_sum_abs = std::max(max_sum_abs, std::fabs(static_cast<double>(got)));
      }
    }
  }
  const double mean_ms = total_ms / repeats;
  csv << mode_name << ',' << a.rows << ',' << a.cols << ',' << a.values.size() << ',' << block_rows << ',' << blocks << ',' << repeats << ','
      << std::setprecision(12) << mean_ms << ',' << max_abs << ',' << max_rel << ',' << max_sum_abs << '\n';
  std::printf("%s rows=%d cols=%d block_rows=%d blocks=%d mean_ms=%.6f max_abs=%.9Le max_rel=%.9Le max_abs_output=%.9Le\n",
              mode_name.c_str(), a.rows, a.cols, block_rows, blocks, mean_ms,
              static_cast<long double>(max_abs), static_cast<long double>(max_rel), static_cast<long double>(max_sum_abs));
  cudaFree(d_treed); cudaFree(d_pd); cudaFree(d_treef); cudaFree(d_pf); cudaFree(d_q); cudaFree(d_values); cudaFree(d_columns); cudaFree(d_offsets);
  return 0;
}

static CsrMatrix make_input(int rows, int cols, int block_rows) {
  CsrMatrix a; a.rows = rows; a.cols = cols; a.offsets.resize(rows + 1);
  a.columns.reserve(static_cast<size_t>(rows) * cols);
  a.values.reserve(static_cast<size_t>(rows) * cols);
  for (int r = 0; r < rows; ++r) {
    const int in_block = r % block_rows;
    for (int c = 0; c < cols; ++c) {
      a.columns.push_back(c);
      // 1e6 products reach a range where FP32 ulps exceed the residual 1.
      // The zero row keeps the positive/negative large terms balanced.
      const float v = (in_block < block_rows / 2 - 1) ? 1000000.0F
                    : (in_block == block_rows / 2 - 1 ? 1.0F
                    : (in_block < block_rows - 1 ? -1000000.0F : 0.0F));
      a.values.push_back(v);
    }
    a.offsets[r + 1] = static_cast<int>(a.values.size());
  }
  return a;
}

int main(int argc, char** argv) {
  const int rows = argc > 1 ? std::atoi(argv[1]) : 65536;
  const int cols = argc > 2 ? std::atoi(argv[2]) : 8;
  const int block_rows = argc > 3 ? std::atoi(argv[3]) : 256;
  const int repeats = argc > 4 ? std::atoi(argv[4]) : 5;
  const std::string output = argc > 5 ? argv[5] : "accumulation_stress.csv";
  if (rows <= 0 || cols <= 0 || block_rows <= 1 || repeats <= 0) return 64;
  CsrMatrix a = make_input(rows, cols, block_rows);
  const int l = 2;
  std::vector<float> q(static_cast<size_t>(rows) * l, 1.0F);
  std::vector<long double> ref(static_cast<size_t>(cols) * l, 0.0L);
  for (int r = 0; r < rows; ++r)
    for (int j = a.offsets[r]; j < a.offsets[r + 1]; ++j)
      for (int k = 0; k < l; ++k) ref[a.columns[j] + static_cast<size_t>(k) * cols] +=
          static_cast<long double>(a.values[j]) * static_cast<long double>(q[r + static_cast<size_t>(k) * rows]);
  std::ofstream csv(output);
  if (!csv) return 66;
  csv << "mode,rows,cols,nnz,block_rows,blocks,repeats,mean_ms,max_abs_error,max_relative_error,max_abs_output\n";
  if (run_mode<AccumMode::Fp32>(a, q, block_rows, repeats, "fp32", csv, ref) != 0) return 1;
  if (run_mode<AccumMode::Fp64>(a, q, block_rows, repeats, "fp64", csv, ref) != 0) return 1;
  if (run_mode<AccumMode::Kahan>(a, q, block_rows, repeats, "kahan_fp32", csv, ref) != 0) return 1;
  return 0;
}
