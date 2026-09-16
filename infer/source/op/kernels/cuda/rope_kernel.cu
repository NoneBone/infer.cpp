#include "rope_kernel.cuh"
#include <cuda_runtime.h>
#include <math.h>
namespace kernel
{
 __global__ void rope_fp16(int pos, int dim, int kv_dim, int hs, __half *q, __half *k, const float *sn, const float *cs)
 {
  int idx = blockIdx.x * blockDim.x + threadIdx.x, pairs = dim / 2;
  if (idx >= pairs)
   return;
  int h = idx / (hs / 2), d = idx % (hs / 2), base = h * hs;
  int a = base + d, b = base + d + hs / 2;
  float si = sn[pos * hs + d * 2], co = cs[pos * hs + d * 2];
  __half *v = h * hs < kv_dim ? k : q;
  float x = __half2float(v[a]), y = __half2float(v[b]);
  v[a] = __float2half(co * x - si * y);
  v[b] = __float2half(co * y + si * x);
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
}
