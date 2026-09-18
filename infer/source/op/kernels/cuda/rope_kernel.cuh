#ifndef INFER_OP_CU_KERNEL_ROPE
#define INFER_OP_CU_KERNEL_ROPE
#include "tensor/tensor.h"
namespace kernel {
void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                    const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                    const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache, void* stream);

void sin_cos_cache_calc_cu(int head_size, int max_seq_len, const tensor::Tensor& sin_cache,
                           const tensor::Tensor& cos_cache, cudaStream_t stream);

void batched_rope_kernel_cu(int32_t start_pos, int32_t seq_len, int32_t dim, int32_t kv_dim,
                            int32_t head_size, const tensor::Tensor& input_q,
                            const tensor::Tensor& input_k, const tensor::Tensor& sin_cache,
                            const tensor::Tensor& cos_cache, void* stream);
void batched_kv_cache_kernel_cu(int32_t layer_idx, int32_t start_pos, int32_t seq_len,
                                int32_t max_seq_len, int32_t kv_dim,
                                const tensor::Tensor& key, const tensor::Tensor& value,
                                const tensor::Tensor& key_cache, const tensor::Tensor& value_cache,
                                void* stream);

}  // namespace kernel
#endif  // ROPE_KERNEL_CU_CUH
