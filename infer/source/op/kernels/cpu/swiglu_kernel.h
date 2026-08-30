#ifndef INFER_OP_CPU_KERNEL_SWIGLU
#define INFER_OP_CPU_KERNEL_SWIGLU
#include "tensor/tensor.h"
namespace kernel {
void swiglu_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& output, void* stream);
}  // namespace kernel
#endif