#ifndef KUIPER_INCLUDE_FLUX_TEXT_LAYERS_H_
#define KUIPER_INCLUDE_FLUX_TEXT_LAYERS_H_
#include "base/base.h"
#include "tensor/tensor.h"
namespace flux {
base::Status embedding(const tensor::Tensor& token_ids, const tensor::Tensor& token_weight,
                       const tensor::Tensor* position_weight, tensor::Tensor& output);
base::Status linear(const tensor::Tensor& input, const tensor::Tensor& weight,
                    const tensor::Tensor* bias, tensor::Tensor& output);
base::Status residual_add(const tensor::Tensor& input, const tensor::Tensor& residual,
                          tensor::Tensor& output);
base::Status self_attention(const tensor::Tensor& input, const tensor::Tensor& q_weight,
                            const tensor::Tensor& k_weight, const tensor::Tensor& v_weight,
                            const tensor::Tensor& o_weight, int32_t head_count, bool causal,
                            tensor::Tensor& output, const tensor::Tensor* attention_bias = nullptr);
int32_t t5_relative_position_bucket(int32_t relative_position, bool bidirectional,
                                    int32_t num_buckets = 32, int32_t max_distance = 128);
base::Status t5_relative_position_bias(const tensor::Tensor& relative_attention_bias, int32_t seq,
                                       bool bidirectional, tensor::Tensor& output);
base::Status clip_causal_mask(int32_t seq, tensor::Tensor& output);
base::Status clip_encoder_block(const tensor::Tensor& input, const tensor::Tensor& norm1_weight,
                                const tensor::Tensor& norm1_bias, const tensor::Tensor& q_weight,
                                const tensor::Tensor& q_bias, const tensor::Tensor& k_weight,
                                const tensor::Tensor& k_bias, const tensor::Tensor& v_weight,
                                const tensor::Tensor& v_bias, const tensor::Tensor& o_weight,
                                const tensor::Tensor& o_bias, const tensor::Tensor& norm2_weight,
                                const tensor::Tensor& norm2_bias, const tensor::Tensor& fc1_weight,
                                const tensor::Tensor& fc1_bias, const tensor::Tensor& fc2_weight,
                                const tensor::Tensor& fc2_bias, int32_t head_count,
                                tensor::Tensor& output);
base::Status clip_mlp(const tensor::Tensor& input, const tensor::Tensor& fc1_weight,
                      const tensor::Tensor& fc1_bias, const tensor::Tensor& fc2_weight,
                      const tensor::Tensor& fc2_bias, tensor::Tensor& output);
base::Status t5_gated_mlp(const tensor::Tensor& input, const tensor::Tensor& wi0_weight,
                          const tensor::Tensor& wi1_weight, const tensor::Tensor& wo_weight,
                          tensor::Tensor& output);
}  // namespace flux
#endif
