#ifndef INFER_SOURCE_OP_KERNELS_CPU_SCALE_SUM_KERNEL_H_
#define INFER_SOURCE_OP_KERNELS_CPU_SCALE_SUM_KERNEL_H_

#include "tensor/tensor.h"

namespace kernel {

void scale_sum_kernel_cpu(const tensor::Tensor& value, const tensor::Tensor& scale,
                          const tensor::Tensor& output, int t, int size, int stride,
                          void* stream);

}  // namespace kernel

#endif
