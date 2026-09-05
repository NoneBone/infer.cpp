#include "scale_sum_kernel.cuh"

#include "base/base.h"

namespace kernel {
__global__ void scale_sum_kernel_impl(const float* value, const float* scale, float* output,
                                      int t, int size, int stride) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  float sum = 0.f;
  for (int j = 0; j <= t; ++j) {
    sum += value[j * stride + idx] * scale[j];
  }
  output[idx] = sum;
}

void scale_sum_kernel_cu(const tensor::Tensor& value, const tensor::Tensor& scale,
                         const tensor::Tensor& output, int t, int size, int stride,
                         void* stream) {
  CHECK(!value.is_empty());
  CHECK(!scale.is_empty());
  CHECK(!output.is_empty());
  CHECK(value.data_type() == base::DataType::kDataTypeFp32);
  CHECK(scale.data_type() == base::DataType::kDataTypeFp32);
  CHECK(output.data_type() == base::DataType::kDataTypeFp32);

  const float* value_ptr = value.ptr<float>();
  const float* scale_ptr = scale.ptr<float>();
  float* output_ptr = const_cast<float*>(output.ptr<float>());

  const int thread_num = 256;
  const int block_num = CEIL(size, thread_num);
  cudaStream_t stream_ = reinterpret_cast<cudaStream_t>(stream);
  if (stream_) {
    scale_sum_kernel_impl<<<block_num, thread_num, 0, stream_>>>(value_ptr, scale_ptr, output_ptr,
                                                                 t, size, stride);
  } else {
    scale_sum_kernel_impl<<<block_num, thread_num>>>(value_ptr, scale_ptr, output_ptr, t, size,
                                                     stride);
  }
}

}  // namespace kernel
