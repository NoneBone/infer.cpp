#include "mha_kernel.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
namespace kernel
{
 __global__ void mha_fp16(int pos, int seq, int kvdim, int kvmul, int heads, int hs, int off, const __half *q, __half *score, __half *out, const __half *kc, const __half *vc)
 {
  int h = blockIdx.x;
  if (h >= heads)
   return;
  int ho = (h / kvmul) * hs;
  extern __shared__ float sq[];
  for (int i = threadIdx.x; i < hs; i += blockDim.x)
   sq[i] = __half2float(q[h * hs + i]);
  __syncthreads();
  for (int t = threadIdx.x; t <= pos; t += blockDim.x)
  {
   float s = 0;
   for (int i = 0; i < hs; i++)
    s += sq[i] * __half2float(kc[off + t * kvdim + ho + i]);
   score[h * seq + t] = __float2half(s / sqrtf((float)hs));
  }
  __syncthreads();
  if (threadIdx.x == 0)
  {
   float mx = -1e30f, sum = 0;
   for (int t = 0; t <= pos; t++)
    mx = fmaxf(mx, __half2float(score[h * seq + t]));
   for (int t = 0; t <= pos; t++)
   {
    float e = expf(__half2float(score[h * seq + t]) - mx);
    score[h * seq + t] = __float2half(e);
    sum += e;
   }
   for (int t = 0; t <= pos; t++)
    score[h * seq + t] = __float2half(__half2float(score[h * seq + t]) / sum);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < hs; i += blockDim.x)
  {
   float v = 0;
   for (int t = 0; t <= pos; t++)
    v += __half2float(score[h * seq + t]) * __half2float(vc[off + t * kvdim + ho + i]);
   out[h * hs + i] = __float2half(v);
  }
 }
 void mha_kernel_cu(int32_t pos, int32_t heads, int32_t layer, int32_t seq, int32_t kvdim, int32_t kvmul, int32_t hs, const tensor::Tensor &out, const tensor::Tensor &q, const tensor::Tensor &score, const tensor::Tensor &kc, const tensor::Tensor &vc, base::DeviceType, CudaConfig *cfg)
 {
  CHECK(cfg);
  CHECK(out.data_type() == base::DataType::kDataTypeFp16);
  int off = layer * seq * kvdim;
  mha_fp16<<<heads, 256, hs * sizeof(float), cfg->stream>>>(pos, seq, kvdim, kvmul, heads, hs, off, q.ptr<__half>(), const_cast<__half *>(score.ptr<__half>()), const_cast<__half *>(out.ptr<__half>()), kc.ptr<__half>(), vc.ptr<__half>());
 }
}
