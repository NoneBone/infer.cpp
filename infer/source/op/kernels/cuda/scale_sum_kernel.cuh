#ifndef INFER_SOURCE_OP_KERNELS_CUDA_SCALE_SUM_KERNEL_CUH_
#define INFER_SOURCE_OP_KERNELS_CUDA_SCALE_SUM_KERNEL_CUH_

#include "tensor/tensor.h"

namespace kernel {

void scale_sum_kernel_cu(const tensor::Tensor& value, const tensor::Tensor& scale,
                         const tensor::Tensor& output, int t, int size, int stride,
                         void* stream);

}  // namespace kernel

#endif
