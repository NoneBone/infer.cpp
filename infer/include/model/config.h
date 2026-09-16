#ifndef INFER_INCLUDE_MODEL_CONFIG_H_
#define INFER_INCLUDE_MODEL_CONFIG_H_
// #include <unistd.h>
#include "cuda_fp16.h"
#include <cstddef>
#include <cstdint>
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
struct RawModelData {
  ~RawModelData();
  int32_t fd = -1;
  size_t file_size = 0;
  void* data = nullptr;
  void* weight_data = nullptr;

  virtual const void* weight(size_t offset) const = 0;
};

struct RawModelDataFp32 : RawModelData {
  const void* weight(size_t offset) const override;
};
struct RawModelDataFp16 : RawModelData {
  const void* weight(size_t offset) const override;
};
struct RawModelDataInt8 : RawModelData {
  const void* weight(size_t offset) const override;
};
} // namespace model
struct TransformerConfig {
  int32_t kv_dim_ = 0;
  int32_t kv_mul_ = 0;
  int32_t head_size_ = 0;
  int32_t vocab_size_ = 0;

  int32_t dim_ = 0;
  int32_t hidden_dim_ = 0;
  int32_t layer_num_ = 0;
  int32_t head_num_ = 0;
  int32_t kv_head_num_ = 0;
  int32_t seq_len_ = 0;
  bool is_shared_weight_ = false;
  int32_t immediate_dim_ = 0;
};

#endif