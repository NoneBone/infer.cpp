#include <tensor/tensor.h>
#include "swiglu_kernel.cuh"
namespace kernel {
__global__ void swiglu_kernel_cu_fp16(int size, const __half* in1, const __half* in2, __half* out) {
  int tid = threadIdx.x;
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) {
    return;
  }
  extern __shared__ __half shared_mem[];
  __half* smem1 = shared_mem;
  __half* smem2 = shared_mem + blockDim.x;

  smem1[tid] = in1[idx];
  smem2[tid] = in2[idx];
  __syncthreads();

  float x = __half2float(smem1[tid]);
  float value = 1.0f / (1.0f + expf(-x));
  smem1[tid] = __float2half(x * value);

  out[idx] = smem1[tid] * smem2[tid];
}

void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                      const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK(input1.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK_EQ(input2.is_empty(), false);
  CHECK(input2.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK_EQ(output.is_empty(), false);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  int size = static_cast<int32_t>(input1.size());
  int threads = 128;
  int blocks = (size + threads - 1) / threads;
  const size_t shmem = threads * sizeof(__half) * 2;
  if (!stream) {
    swiglu_kernel_cu_fp16<<<blocks, threads, shmem>>>(
        size, input1.ptr<__half>(), input2.ptr<__half>(), const_cast<__half*>(output.ptr<__half>()));
  } else {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    swiglu_kernel_cu_fp16<<<blocks, threads, shmem, stream_>>>(
        size, input1.ptr<__half>(), input2.ptr<__half>(), const_cast<__half*>(output.ptr<__half>()));
  }
}
}
