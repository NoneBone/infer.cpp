#ifndef INFER_OP_CU_KERNEL_SOFTMAX
#define INFER_OP_CU_KERNEL_SOFTMAX
#include "tensor/tensor.h"
namespace kernel {
void softmax_inplace_cu(const tensor::Tensor& input, void* stream);
}  // namespace kernel
#endif  // ROPE_KERNEL_CU_CUH
