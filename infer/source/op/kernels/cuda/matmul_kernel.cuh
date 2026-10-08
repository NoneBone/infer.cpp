#ifndef INFER_OP_CU_KERNEL_MAT
#define INFER_OP_CU_KERNEL_MAT
#ifndef INFER_MATVEC_FP16_BACKEND
#define INFER_MATVEC_FP16_BACKEND 1
#endif
#define BATCHED_MATMUL_FP16_BACKEND 1

#include "tensor/tensor.h"
#include "base/cuda_config.h"
namespace kernel {
void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, float scale = 1.f,
                      const CudaConfig* config = nullptr);
void batched_matmul_kernel_cu_pure_fp16(const tensor::Tensor& input, const tensor::Tensor& weight,
                                         const tensor::Tensor& output, int32_t batch_size, 
                                         float scale, const CudaConfig* config);
}  // namespace kernel
#endif
