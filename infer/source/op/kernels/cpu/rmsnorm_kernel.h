#ifndef INFER_OP_CPU_KERNEL_RMS
#define INFER_OP_CPU_KERNEL_RMS
#include "base/cuda_config.h"
#include "tensor/tensor.h"
namespace kernel {
void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& output, void* stream = nullptr);
}  // namespace kernel
#endif