#ifndef INFER_OP_CU_KERNEL_RMS
#define INFER_OP_CU_KERNEL_RMS
#include "tensor/tensor.h"
namespace kernel {
void rmsnorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, void* stream = nullptr);

void rmsnorm_kernel_cu_dim(const tensor::Tensor& input, const tensor::Tensor& weight,
                           const tensor::Tensor& output, int32_t dim, void* stream = nullptr);
}  // namespace kernel
#endif