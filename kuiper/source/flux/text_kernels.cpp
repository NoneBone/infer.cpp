#include "flux/text_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace flux {
void layer_norm_cuda(const tensor::Tensor&, const tensor::Tensor&, const tensor::Tensor&, tensor::Tensor&, float);
void rms_norm_cuda(const tensor::Tensor&, const tensor::Tensor&, tensor::Tensor&, float);
void quick_gelu_cuda(const tensor::Tensor&, tensor::Tensor&);
void gelu_new_cuda(const tensor::Tensor&, tensor::Tensor&);
void softmax_cuda(const tensor::Tensor&, tensor::Tensor&);
void gated_gelu_cuda(const tensor::Tensor&, const tensor::Tensor&, tensor::Tensor&);
float bf16_to_float(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}
uint16_t float_to_bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
namespace {
base::Status check(const tensor::Tensor& input, const tensor::Tensor& output) {
  if (input.data_type() != base::DataType::kDataTypeBf16 ||
      output.data_type() != base::DataType::kDataTypeBf16 || input.dims() != output.dims()) {
    return base::error::InvalidArgument("text kernel requires matching CPU BF16 tensors");
  }
  return base::error::Success();
}
}

base::Status layer_norm(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& bias, tensor::Tensor& output, float epsilon) {
  auto status = check(input, output);
  if (!status || weight.dims_size() != 1 || bias.dims() != weight.dims() ||
      input.size() % weight.size() != 0) return base::error::InvalidArgument("invalid LayerNorm shape");
  if (input.device_type() == base::DeviceType::kDeviceCUDA) { layer_norm_cuda(input, weight, bias, output, epsilon); return base::error::Success(); }
  const size_t dim = weight.size();
  const auto* x = input.ptr<uint16_t>(); const auto* w = weight.ptr<uint16_t>();
  const auto* b = bias.ptr<uint16_t>(); auto* y = output.ptr<uint16_t>();
  for (size_t row = 0; row < input.size() / dim; ++row) {
    float mean = 0.f, variance = 0.f;
    for (size_t i = 0; i < dim; ++i) mean += bf16_to_float(x[row * dim + i]);
    mean /= dim;
    for (size_t i = 0; i < dim; ++i) { float d = bf16_to_float(x[row * dim + i]) - mean; variance += d * d; }
    const float inv_std = 1.f / std::sqrt(variance / dim + epsilon);
    for (size_t i = 0; i < dim; ++i)
      y[row * dim + i] = float_to_bf16((bf16_to_float(x[row * dim + i]) - mean) * inv_std *
                                        bf16_to_float(w[i]) + bf16_to_float(b[i]));
  }
  return base::error::Success();
}

base::Status rms_norm(const tensor::Tensor& input, const tensor::Tensor& weight,
                      tensor::Tensor& output, float epsilon) {
  auto status = check(input, output);
  if (!status || weight.dims_size() != 1 || input.size() % weight.size() != 0)
    return base::error::InvalidArgument("invalid RMSNorm shape");
  if (input.device_type() == base::DeviceType::kDeviceCUDA) { rms_norm_cuda(input, weight, output, epsilon); return base::error::Success(); }
  const size_t dim = weight.size(); const auto* x = input.ptr<uint16_t>();
  const auto* w = weight.ptr<uint16_t>(); auto* y = output.ptr<uint16_t>();
  for (size_t row = 0; row < input.size() / dim; ++row) {
    float square_mean = 0.f;
    for (size_t i = 0; i < dim; ++i) { float v = bf16_to_float(x[row * dim + i]); square_mean += v * v; }
    const float inv_rms = 1.f / std::sqrt(square_mean / dim + epsilon);
    for (size_t i = 0; i < dim; ++i)
      y[row * dim + i] = float_to_bf16(bf16_to_float(x[row * dim + i]) * inv_rms * bf16_to_float(w[i]));
  }
  return base::error::Success();
}

base::Status quick_gelu(const tensor::Tensor& input, tensor::Tensor& output) {
  auto status = check(input, output); if (!status) return status;
  if (input.device_type() == base::DeviceType::kDeviceCUDA) { quick_gelu_cuda(input, output); return base::error::Success(); }
  for (size_t i = 0; i < input.size(); ++i) { float x = bf16_to_float(input.ptr<uint16_t>()[i]); output.ptr<uint16_t>()[i] = float_to_bf16(x / (1.f + std::exp(-1.702f * x))); }
  return base::error::Success();
}
base::Status gelu_new(const tensor::Tensor& input, tensor::Tensor& output) {
  auto status = check(input, output); if (!status) return status;
  if (input.device_type() == base::DeviceType::kDeviceCUDA) { gelu_new_cuda(input, output); return base::error::Success(); }
  constexpr float k = 0.7978845608f;
  for (size_t i = 0; i < input.size(); ++i) { float x = bf16_to_float(input.ptr<uint16_t>()[i]); output.ptr<uint16_t>()[i] = float_to_bf16(0.5f * x * (1.f + std::tanh(k * (x + 0.044715f * x * x * x)))); }
  return base::error::Success();
}
base::Status softmax(const tensor::Tensor& input, tensor::Tensor& output) {
  auto status = check(input, output); if (!status || input.dims_size() == 0) return base::error::InvalidArgument("invalid softmax tensor");
  if (input.device_type() == base::DeviceType::kDeviceCUDA) { softmax_cuda(input, output); return base::error::Success(); }
  const size_t dim = input.get_dim(input.dims_size() - 1);
  for (size_t row = 0; row < input.size() / dim; ++row) {
    float maximum = -INFINITY, sum = 0.f;
    for (size_t i = 0; i < dim; ++i) maximum = std::max(maximum, bf16_to_float(input.ptr<uint16_t>()[row * dim + i]));
    for (size_t i = 0; i < dim; ++i) sum += std::exp(bf16_to_float(input.ptr<uint16_t>()[row * dim + i]) - maximum);
    for (size_t i = 0; i < dim; ++i) output.ptr<uint16_t>()[row * dim + i] = float_to_bf16(std::exp(bf16_to_float(input.ptr<uint16_t>()[row * dim + i]) - maximum) / sum);
  }
  return base::error::Success();
}
base::Status gated_gelu(const tensor::Tensor& value, const tensor::Tensor& gate, tensor::Tensor& output) {
  auto status = check(value, output);
  if (!status || gate.dims() != value.dims() || gate.data_type() != base::DataType::kDataTypeBf16)
    return base::error::InvalidArgument("invalid gated GELU shape");
  if (value.device_type() == base::DeviceType::kDeviceCUDA) { gated_gelu_cuda(value, gate, output); return base::error::Success(); }
  tensor::Tensor activated(base::DataType::kDataTypeBf16, value.dims(), true, base::CPUDeviceAllocatorFactory::get_instance());
  status = gelu_new(gate, activated); if (!status) return status;
  for (size_t i = 0; i < value.size(); ++i) output.ptr<uint16_t>()[i] = float_to_bf16(bf16_to_float(value.ptr<uint16_t>()[i]) * bf16_to_float(activated.ptr<uint16_t>()[i]));
  return base::error::Success();
}
}  // namespace flux
