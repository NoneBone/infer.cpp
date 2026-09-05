#ifndef INFER_INCLUDE_MODEL_CONFIG_H_
#define INFER_INCLUDE_MODEL_CONFIG_H_

namespace model
{
struct ModelConfig
{
    int32_t dim = 0;
    int32_t hidden_dim = 0;
    int32_t layer_num = 0;
    int32_t head_num = 0;
    int32_t kv_head_num = 0;
    int32_t vocab_size = 0;
    int32_t seq_len = 0;
    int32_t immediate_dim_ = 0;
};

} // namespace model


#endif