#ifndef INFER_OP_CU_KERNEL_SWIGLU
#define INFER_OP_CU_KERNEL_SWIGLU
#include "tensor/tensor.h"
namespace kernel {
void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                      const tensor::Tensor& output, void* stream);
}  // namespace kernel
#endif  // ROPE_KERNEL_CU_CUH
