#include "argmax_kernel.cuh"
#include <cuda_runtime.h>
namespace kernel
{
 __global__ void argmax_fp16(const __half *x, size_t n, size_t *out)
 {
  if (threadIdx.x == 0)
  {
   size_t best = 0;
   float v = -1e30f;
   for (size_t i = 0; i < n; i++)
   {
    float z = __half2float(x[i]);
    if (z > v)
    {
     v = z;
     best = i;
    }
   }
   *out = best;
  }
 }
 size_t argmax_kernel_cu(const __half *x, size_t n, void *stream)
 {
  size_t *d = nullptr;
  cudaMalloc(&d, sizeof(size_t));
  size_t h = 0;
  auto s = static_cast<cudaStream_t>(stream);
  argmax_fp16<<<1, 1, 0, s>>>(x, n, d);
  cudaMemcpyAsync(&h, d, sizeof(size_t), cudaMemcpyDeviceToHost, s);
  cudaStreamSynchronize(s);
  cudaFree(d);
  return h;
 }
}
