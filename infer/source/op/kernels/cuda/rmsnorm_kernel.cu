#include "rmsnorm_kernel.cuh"
#include <cuda_runtime.h>
namespace kernel
{
  __global__ void rmsnorm_fp16(const __half *in, const __half *w, __half *out, int rows, int n, float eps)
  {
    int row = blockIdx.x;
    if (row >= rows)
      return;
    __shared__ float p[128];
    float s = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
      float x = __half2float(in[row * n + i]);
      s += x * x;
    }
    p[threadIdx.x] = s;
    __syncthreads();
    for (int d = 64; d; d >>= 1)
    {
      if (threadIdx.x < d)
        p[threadIdx.x] += p[threadIdx.x + d];
      __syncthreads();
    }
    float scale = rsqrtf(p[0] / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
      out[row * n + i] = __float2half(__half2float(in[row * n + i]) * __half2float(w[i]) * scale);
  }
  static void launch(const tensor::Tensor &in, const tensor::Tensor &w, const tensor::Tensor &out, int rows, int n, void *stream)
  {
    CHECK(in.data_type() == base::DataType::kDataTypeFp16);
    CHECK(w.data_type() == base::DataType::kDataTypeFp16);
    CHECK(out.data_type() == base::DataType::kDataTypeFp16);
    rmsnorm_fp16<<<rows, 128, 0, static_cast<cudaStream_t>(stream)>>>(in.ptr<__half>(), w.ptr<__half>(), const_cast<__half *>(out.ptr<__half>()), rows, n, 1e-6f);
  }
  void rmsnorm_kernel_cu(const tensor::Tensor &in, const tensor::Tensor &w, const tensor::Tensor &out, void *stream) { launch(in, w, out, 1, in.size(), stream); }
  void rmsnorm_kernel_cu_dim(const tensor::Tensor &in, const tensor::Tensor &w, const tensor::Tensor &out, int32_t, void *stream)
  {
    int n = in.get_dim(in.dims_size() - 1);
    launch(in, w, out, in.size() / n, n, stream);
  }
}
