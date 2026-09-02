#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include "flux/text_kernels.h"

namespace flux {
namespace {
constexpr int kThreads = 256;
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 to_bf16(float x) { return __float2bfloat16(x); }
template <int T>
__global__ void norm_kernel(const __nv_bfloat16* x, const __nv_bfloat16* w, const __nv_bfloat16* b,
                            __nv_bfloat16* y, int rows, int dim, float eps, bool rms) {
  int row = blockIdx.x;
  if (row >= rows) return;
  float sum = 0.f, sq = 0.f;
  for (int i = threadIdx.x; i < dim; i += T) {
    float v = to_float(x[row * dim + i]);
    sum += v;
    sq += v * v;
  }
  using R = cub::BlockReduce<float, T>;
  __shared__ typename R::TempStorage store;
  __shared__ float mean, scale;
  sum = R(store).Sum(sum);
  __syncthreads();
  if (threadIdx.x == 0) mean = rms ? 0.f : sum / dim;
  __syncthreads();
  sq = R(store).Sum(sq);
  __syncthreads();
  if (threadIdx.x == 0) scale = rsqrtf((rms ? sq : sq - dim * mean * mean) / dim + eps);
  __syncthreads();
  for (int i = threadIdx.x; i < dim; i += T)
    y[row * dim + i] = to_bf16((to_float(x[row * dim + i]) - mean) * scale * to_float(w[i]) +
                               (b ? to_float(b[i]) : 0.f));
}
__global__ void point_kernel(const __nv_bfloat16* x, __nv_bfloat16* y, int n, int kind) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = to_float(x[i]);
  float z = kind ? 0.5f * v * (1.f + tanhf(.7978845608f * (v + .044715f * v * v * v)))
                 : v / (1.f + expf(-1.702f * v));
  y[i] = to_bf16(z);
}
template <int T>
__global__ void softmax_kernel(const __nv_bfloat16* x, __nv_bfloat16* y, int rows, int dim) {
  int row = blockIdx.x;
  if (row >= rows) return;
  float m = -INFINITY;
  for (int i = threadIdx.x; i < dim; i += T) m = fmaxf(m, to_float(x[row * dim + i]));
  using R = cub::BlockReduce<float, T>;
  __shared__ typename R::TempStorage store;
  __shared__ float maxv, den;
  m = R(store).Reduce(m, cub::Max());
  __syncthreads();
  if (threadIdx.x == 0) maxv = m;
  __syncthreads();
  float s = 0;
  for (int i = threadIdx.x; i < dim; i += T) s += expf(to_float(x[row * dim + i]) - maxv);
  s = R(store).Sum(s);
  __syncthreads();
  if (threadIdx.x == 0) den = s;
  __syncthreads();
  for (int i = threadIdx.x; i < dim; i += T)
    y[row * dim + i] = to_bf16(expf(to_float(x[row * dim + i]) - maxv) / den);
}
__global__ void gated_kernel(const __nv_bfloat16* v, const __nv_bfloat16* g, __nv_bfloat16* y,
                             int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = to_float(g[i]);
  y[i] =
      to_bf16(to_float(v[i]) * .5f * x * (1.f + tanhf(.7978845608f * (x + .044715f * x * x * x))));
}
const __nv_bfloat16* p(const tensor::Tensor& t) {
  return reinterpret_cast<const __nv_bfloat16*>(t.ptr<uint16_t>());
}
__nv_bfloat16* p(tensor::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.ptr<uint16_t>()); }
int rows(const tensor::Tensor& t) { return t.size() / t.get_dim(t.dims_size() - 1); }
void check() { CHECK_EQ(cudaGetLastError(), cudaSuccess); }
}  // namespace
void layer_norm_cuda(const tensor::Tensor& i, const tensor::Tensor& w, const tensor::Tensor& b,
                     tensor::Tensor& o, float e) {
  norm_kernel<kThreads><<<rows(i), kThreads>>>(p(i), p(w), p(b), p(o), rows(i), w.size(), e, false);
  check();
}
void rms_norm_cuda(const tensor::Tensor& i, const tensor::Tensor& w, tensor::Tensor& o, float e) {
  norm_kernel<kThreads>
      <<<rows(i), kThreads>>>(p(i), p(w), nullptr, p(o), rows(i), w.size(), e, true);
  check();
}
void quick_gelu_cuda(const tensor::Tensor& i, tensor::Tensor& o) {
  point_kernel<<<(i.size() + kThreads - 1) / kThreads, kThreads>>>(p(i), p(o), i.size(), 0);
  check();
}
void gelu_new_cuda(const tensor::Tensor& i, tensor::Tensor& o) {
  point_kernel<<<(i.size() + kThreads - 1) / kThreads, kThreads>>>(p(i), p(o), i.size(), 1);
  check();
}
void softmax_cuda(const tensor::Tensor& i, tensor::Tensor& o) {
  softmax_kernel<kThreads>
      <<<rows(i), kThreads>>>(p(i), p(o), rows(i), i.get_dim(i.dims_size() - 1));
  check();
}
void gated_gelu_cuda(const tensor::Tensor& v, const tensor::Tensor& g, tensor::Tensor& o) {
  gated_kernel<<<(v.size() + kThreads - 1) / kThreads, kThreads>>>(p(v), p(g), p(o), v.size());
  check();
}
}  // namespace flux
