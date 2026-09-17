#ifndef INFER_INCLUDE_MODEL_QWEN3_H_
#define INFER_INCLUDE_MODEL_QWEN3_H_

#include "op/layer.h"
#include "model/model.h"

namespace model {
struct QWen3TransformerConfig {
  int32_t kv_dim_ = 0;
  int32_t kv_mul_ = 0;
  int32_t head_size_ = 0;
  int32_t immediate_size_ = 0;
  int32_t vocab_size_ = 0;

  int32_t dim_ = 0;
  int32_t hidden_dim_ = 0;
  int32_t layer_num_ = 0;
  int32_t head_num_ = 0;
  int32_t kv_head_num_ = 0;
  int32_t seq_len_ = 0;
  bool is_shared_weight_ = false;
};

struct Qwen3Layers{
    std::shared_ptr<op::Layer> add_layer_;
    std::shared_ptr<op::Layer> rope_layer_;
    std::shared_ptr<op::Layer> swiglu_layer_;
    std::shared_ptr<op::Layer> mha_layer_;
    
    std::vector<std::shared_ptr<op::Layer>> wq_layers_;
    std::vector<std::shared_ptr<op::Layer>> wk_layers_;
    std::vector<std::shared_ptr<op::Layer>> wv_layers_;
    std::vector<std::shared_ptr<op::Layer>> wo_layers_;

    std::vector<std::shared_ptr<op::Layer>> w1_layers_;
    std::vector<std::shared_ptr<op::Layer>> w2_layers_;
    std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;
    std::vector<std::shared_ptr<op::Layer>> w3_layers_;

    std::shared_ptr<op::Layer> cls_layer_;
    std::shared_ptr<op::Layer> emb_layer_;

    std::shared_ptr<op::BatchedMatmulHelperLayer> batched_matmul_helper_layer_;
    void to_cuda(std::shared_ptr<kernel::CudaConfig> config);
};

class Qwen3Model : public Model{
public:
    explicit Qwen3Model(base::TokenizerType tokenizer_type, std::string token_path, std::string model_path, bool is_quant);

    base::Status init(base::DeviceType device_type) override;

    base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                        bool is_prompt, int& next) const override;

    base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                        int& next) const override;

    op::EmbeddingOutput embedding(const std::vector<int>& tokens) const override;
 private:
    void init_mem() override;
    base::Status create_layers() override;

    void create_param_layers() override;
    void create_param_layers_fp16();

    void create_nonparam_layers() override;

    void create_param_quant_layers() override;

    void attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

    void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;

    void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;

    void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;
    
    void batched_attention_qkv(int32_t layer_idx, const tensor::Tensor& rms_out,
                                const tensor::Tensor& query_out, const tensor::Tensor& key_out,
                                const tensor::Tensor& value_out,
                                int32_t seq_len, int32_t start_pos) const;
                                
    void cls_logits(const tensor::Tensor& input) const;

    int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const override;
private:
    std::shared_ptr<kernel::CudaConfig> cuda_config_;
    std::unique_ptr<Qwen3Layers> qwen_layers_;
};
} // model

#endif
