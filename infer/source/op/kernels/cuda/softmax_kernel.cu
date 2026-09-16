#include "softmax_kernel.cuh"
#include <cuda_runtime.h>
#include <math.h>
namespace kernel
{
 __global__ void softmax_fp16(__half *x, int n)
 {
  __shared__ float buf[1024];
  float m = -1e30f, s = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x)
   m = fmaxf(m, __half2float(x[i]));
  buf[threadIdx.x] = m;
  __syncthreads();
  for (int d = blockDim.x / 2; d; d >>= 1)
  {
   if (threadIdx.x < d)
    buf[threadIdx.x] = fmaxf(buf[threadIdx.x], buf[threadIdx.x + d]);
   __syncthreads();
  }
  m = buf[0];
  for (int i = threadIdx.x; i < n; i += blockDim.x)
  {
   float e = expf(__half2float(x[i]) - m);
   x[i] = __float2half(e);
   s += e;
  }
  buf[threadIdx.x] = s;
  __syncthreads();
  for (int d = blockDim.x / 2; d; d >>= 1)
  {
   if (threadIdx.x < d)
    buf[threadIdx.x] += buf[threadIdx.x + d];
   __syncthreads();
  }
  s = buf[0];
  for (int i = threadIdx.x; i < n; i += blockDim.x)
   x[i] = __float2half(__half2float(x[i]) / s);
 }
 void softmax_inplace_cu(const tensor::Tensor &input, void *stream)
 {
  CHECK(!input.is_empty());
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(input.data_type() == base::DataType::kDataTypeFp16);
  int n = input.size();
  softmax_fp16<<<1, 1024, 0, static_cast<cudaStream_t>(stream)>>>(const_cast<__half *>(input.ptr<__half>()), n);
 }
}
