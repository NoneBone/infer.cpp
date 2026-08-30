#ifndef INFER_OP_CU_KERNEL_MAT
#define INFER_OP_CU_KERNEL_MAT
#include "tensor/tensor.h"
#include "base/cuda_config.h"
namespace kernel {
void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, float scale = 1.f,
                      const CudaConfig* config = nullptr);
}  // namespace kernel
#endif