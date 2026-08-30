#ifndef INFER_OP_CU_ADD_KERNEL
#define INFER_OP_CU_ADD_KERNEL
#include "tensor/tensor.h"
namespace kernel {
void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                    const tensor::Tensor& output, void* stream = nullptr);
}  // namespace kernel
#endif