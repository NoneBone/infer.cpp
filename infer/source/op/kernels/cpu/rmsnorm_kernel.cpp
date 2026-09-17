#include "rmsnorm_kernel.h"
#include <cmath>

namespace kernel
{
void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& output, void* stream) {
  UNUSED(stream);
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCPU &&
        weight.device_type() == base::DeviceType::kDeviceCPU &&
        output.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(input.data_type() == base::DataType::kDataTypeFp16);
  CHECK(weight.data_type() == base::DataType::kDataTypeFp16);
  CHECK(output.data_type() == base::DataType::kDataTypeFp16);
  CHECK_EQ(input.dims_size(), 1);
  CHECK_EQ(weight.dims_size(), 1);
  CHECK_EQ(output.dims_size(), 1);
  CHECK_EQ(weight.size(), input.size());
  CHECK_EQ(output.size(), input.size());

  const __half* in_ptr = input.ptr<__half>();
  const __half* wei_ptr = weight.ptr<__half>();
  __half* out_ptr = const_cast<__half*>(output.ptr<__half>());
  const int32_t dim = static_cast<int32_t>(input.size());

  // 使用 FP32 累加提高数值稳定性，最终结果按 FP16 写回。
  float square_sum = 0.f;
  for (int32_t i = 0; i < dim; ++i) {
    const float value = __half2float(in_ptr[i]);
    square_sum += value * value;
  }

  constexpr float eps = 1e-6f;
  const float scale = 1.f / std::sqrt(square_sum / static_cast<float>(dim) + eps);
  for (int32_t i = 0; i < dim; ++i) {
    const float value = __half2float(in_ptr[i]);
    const float norm_weight = __half2float(wei_ptr[i]);
    out_ptr[i] = __float2half(value * norm_weight * scale);
  }
}
} // namespace kernel
