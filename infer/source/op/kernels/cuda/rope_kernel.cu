#include "rope_kernel.cuh"
#include <cuda_runtime.h>
#include <math.h>
namespace kernel
{
 __global__ void rope_fp16(int pos, int dim, int kv_dim, int hs, __half *q, __half *k, const float *sn, const float *cs)
 {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int pairs = dim / 2;
  if (idx >= pairs) return;
  int h = idx / (hs / 2), d = idx % (hs / 2);
  int qbase = h * hs;
  float si = sn[pos * hs + d * 2], co = cs[pos * hs + d * 2];
  float qx = __half2float(q[qbase + d]);
  float qy = __half2float(q[qbase + d + hs / 2]);
  q[qbase + d] = __float2half(co * qx - si * qy);
  q[qbase + d + hs / 2] = __float2half(co * qy + si * qx);
  if (qbase < kv_dim) {
    float kx = __half2float(k[qbase + d]);
    float ky = __half2float(k[qbase + d + hs / 2]);
    k[qbase + d] = __float2half(co * kx - si * ky);
    k[qbase + d + hs / 2] = __float2half(co * ky + si * kx);
  }
 }
 __global__ void rope_cache(int hs, int maxlen, float *sn, float *cs)
 {
  int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= hs)
   return;
  for (int p = 0; p < maxlen; p++)
  {
   float v = p / powf(1000000.f, (float)d / hs);
   sn[p * hs + d] = sinf(v);
   cs[p * hs + d] = cosf(v);
  }
 }
 void sin_cos_cache_calc_cu(int hs, int maxlen, const tensor::Tensor &sn, const tensor::Tensor &cs, cudaStream_t stream)
 {
  int t = 128, b = (hs + t - 1) / t;
  rope_cache<<<b, t, 0, stream>>>(hs, maxlen, const_cast<float *>(sn.ptr<float>()), const_cast<float *>(cs.ptr<float>()));
 }
 void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t hs, const tensor::Tensor &q, const tensor::Tensor &k, const tensor::Tensor &pos, const tensor::Tensor &sn, const tensor::Tensor &cs, void *stream)
 {
  int p = *pos.ptr<int32_t>();
  int t = 128, b = (dim / 2 + t - 1) / t;
  rope_fp16<<<b, t, 0, static_cast<cudaStream_t>(stream)>>>(p, dim, kv_dim, hs, const_cast<__half *>(q.ptr<__half>()), const_cast<__half *>(k.ptr<__half>()), sn.ptr<float>(), cs.ptr<float>());
 }

 __global__ void batched_rope_fp16(int start_pos, int seq_len, int width, int hs,
                                   __half* x, const float* sn, const float* cs) {
  int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= width / 2 || blockIdx.y >= seq_len) return;
  int row = blockIdx.y;
  int head = pair / (hs / 2);
  int d = pair % (hs / 2);
  int base = row * width + head * hs;
  int pos = start_pos + row;
  float s = sn[pos * hs + d * 2];
  float c = cs[pos * hs + d * 2];
  float xv = __half2float(x[base + d]);
  float yv = __half2float(x[base + d + hs / 2]);
  x[base + d] = __float2half(c * xv - s * yv);
  x[base + d + hs / 2] = __float2half(c * yv + s * xv);
 }

 __global__ void copy_batched_kv(int layer, int start_pos, int seq_len, int max_seq, int kv_dim,
                                  const __half* key, const __half* value,
                                  __half* key_cache, __half* value_cache) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= seq_len * kv_dim) return;
  int row = idx / kv_dim;
  int col = idx % kv_dim;
  size_t dst = (static_cast<size_t>(layer) * max_seq + start_pos + row) * kv_dim + col;
  key_cache[dst] = key[idx];
  value_cache[dst] = value[idx];
 }

 void batched_rope_kernel_cu(int32_t start_pos, int32_t seq_len, int32_t dim, int32_t kv_dim,
                             int32_t hs, const tensor::Tensor& q, const tensor::Tensor& k,
                             const tensor::Tensor& sn, const tensor::Tensor& cs, void* stream) {
  cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
  dim3 q_grid((dim / 2 + 127) / 128, seq_len);
  batched_rope_fp16<<<q_grid, 128, 0, stream_>>>(
      start_pos, seq_len, dim, hs, const_cast<__half*>(q.ptr<__half>()),
      sn.ptr<float>(), cs.ptr<float>());
  dim3 k_grid((kv_dim / 2 + 127) / 128, seq_len);
  batched_rope_fp16<<<k_grid, 128, 0, stream_>>>(
      start_pos, seq_len, kv_dim, hs, const_cast<__half*>(k.ptr<__half>()),
      sn.ptr<float>(), cs.ptr<float>());
 }

 void batched_kv_cache_kernel_cu(int32_t layer, int32_t start_pos, int32_t seq_len,
                                 int32_t max_seq, int32_t kv_dim, const tensor::Tensor& key,
                                 const tensor::Tensor& value, const tensor::Tensor& key_cache,
                                 const tensor::Tensor& value_cache, void* stream) {
  int threads = 256;
  int blocks = (seq_len * kv_dim + threads - 1) / threads;
  copy_batched_kv<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
      layer, start_pos, seq_len, max_seq, kv_dim, key.ptr<__half>(), value.ptr<__half>(),
      const_cast<__half*>(key_cache.ptr<__half>()), const_cast<__half*>(value_cache.ptr<__half>()));
 }
}
