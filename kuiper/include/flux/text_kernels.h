#ifndef KUIPER_INCLUDE_FLUX_TEXT_KERNELS_H_
#define KUIPER_INCLUDE_FLUX_TEXT_KERNELS_H_

#include "base/base.h"
#include "tensor/tensor.h"

namespace flux {
float bf16_to_float(uint16_t value);
uint16_t float_to_bf16(float value);

base::Status layer_norm(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& bias, tensor::Tensor& output, float epsilon);
base::Status rms_norm(const tensor::Tensor& input, const tensor::Tensor& weight,
                      tensor::Tensor& output, float epsilon);
base::Status quick_gelu(const tensor::Tensor& input, tensor::Tensor& output);
base::Status gelu_new(const tensor::Tensor& input, tensor::Tensor& output);
base::Status softmax(const tensor::Tensor& input, tensor::Tensor& output);
base::Status gated_gelu(const tensor::Tensor& value, const tensor::Tensor& gate,
                        tensor::Tensor& output);
}  // namespace flux
#endif
