#include "scale_sum_kernel.h"

#include <algorithm>

#include "base/base.h"

namespace kernel {

void scale_sum_kernel_cpu(const tensor::Tensor& value, const tensor::Tensor& scale,
                          const tensor::Tensor& output, int t, int size, int stride,
                          void* stream) {
  UNUSED(stream);
  CHECK(!value.is_empty());
  CHECK(!scale.is_empty());
  CHECK(!output.is_empty());
  CHECK_GE(t, 0);
  CHECK_GT(size, 0);
  CHECK_GT(stride, 0);
  if (value.data_type() != base::DataType::kDataTypeFp32 ||
      scale.data_type() != base::DataType::kDataTypeFp32 ||
      output.data_type() != base::DataType::kDataTypeFp32) {
    LOG(FATAL) << "scale_sum_kernel_cpu only supports fp32 tensors.";
  }

  const float* value_ptr = value.ptr<float>();
  const float* scale_ptr = scale.ptr<float>();
  float* output_ptr = const_cast<float*>(output.ptr<float>());

  for (int i = 0; i < size; ++i) {
    float sum = 0.f;
    for (int j = 0; j <= t; ++j) {
      sum += value_ptr[j * stride + i] * scale_ptr[j];
    }
    output_ptr[i] = sum;
  }
}

}  // namespace kernel
