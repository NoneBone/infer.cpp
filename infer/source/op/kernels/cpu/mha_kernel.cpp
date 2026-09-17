#include "../cpu/mha_kernel.h"
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

namespace kernel {

namespace {

template <typename T>
float to_float(T value) {
  return static_cast<float>(value);
}

template <>
float to_float<__half>(__half value) {
  return __half2float(value);
}

template <typename T>
T from_float(float value) {
  return static_cast<T>(value);
}

template <>
__half from_float<__half>(float value) {
  return __float2half(value);
}

template <typename T>
void mha_kernel_impl(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                     int32_t kv_dim, int32_t kv_mul, int32_t head_size,
                     const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
                     const tensor::Tensor& score_tensor, const tensor::Tensor& key_cache_tensor,
                     const tensor::Tensor& value_cache_tensor) {
  const int32_t layer_offset = layer_index * seq_len * kv_dim;
  const float scale = 1.f / std::sqrt(static_cast<float>(head_size));
  const T* query_ptr = query_tensor.ptr<T>();
  const T* key_ptr = key_cache_tensor.ptr<T>();
  const T* value_ptr = value_cache_tensor.ptr<T>();
  T* score_ptr = const_cast<T*>(score_tensor.ptr<T>());
  T* output_ptr = const_cast<T*>(mha_out.ptr<T>());

  std::vector<float> attention_scores(static_cast<size_t>(pos + 1));
  for (int32_t h = 0; h < head_num; ++h) {
    const int32_t kv_head_offset = (h / kv_mul) * head_size;
    const T* query_head = query_ptr + h * head_size;
    T* score_head = score_ptr + h * seq_len;

    for (int32_t t = 0; t <= pos; ++t) {
      const T* key_head = key_ptr + layer_offset + t * kv_dim + kv_head_offset;
      float dot = 0.f;
      for (int32_t i = 0; i < head_size; ++i) {
        dot += to_float(query_head[i]) * to_float(key_head[i]);
      }
      attention_scores[t] = dot * scale;
    }

    const float max_score =
        *std::max_element(attention_scores.begin(), attention_scores.end());
    float score_sum = 0.f;
    if constexpr (std::is_same_v<T, __half>) {
      // 与 CUDA FP16 内核保持相同的舍入顺序：QK、exp 和归一化结果均写回 FP16。
      for (int32_t t = 0; t <= pos; ++t) {
        score_head[t] = from_float<__half>(attention_scores[t]);
      }
      const float fp16_max_score = to_float(score_head[0]);
      float max_score_fp16 = fp16_max_score;
      for (int32_t t = 1; t <= pos; ++t) {
        max_score_fp16 = std::max(max_score_fp16, to_float(score_head[t]));
      }
      score_sum = 0.f;
      for (int32_t t = 0; t <= pos; ++t) {
        const float e = std::exp(to_float(score_head[t]) - max_score_fp16);
        score_head[t] = from_float<__half>(e);
        score_sum += e;
      }
      for (int32_t t = 0; t <= pos; ++t) {
        score_head[t] = from_float<__half>(to_float(score_head[t]) / score_sum);
        attention_scores[t] = to_float(score_head[t]);
      }
    } else {
      for (int32_t t = 0; t <= pos; ++t) {
        attention_scores[t] = std::exp(attention_scores[t] - max_score);
        score_sum += attention_scores[t];
      }
      for (int32_t t = 0; t <= pos; ++t) {
        attention_scores[t] /= score_sum;
        score_head[t] = from_float<T>(attention_scores[t]);
      }
    }

    T* output_head = output_ptr + h * head_size;
    for (int32_t i = 0; i < head_size; ++i) {
      float value = 0.f;
      for (int32_t t = 0; t <= pos; ++t) {
        const T* value_head = value_ptr + layer_offset + t * kv_dim + kv_head_offset;
        value += attention_scores[t] * to_float(value_head[i]);
      }
      output_head[i] = from_float<T>(value);
    }
  }
}

}  // namespace

void mha_kernel(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len, int32_t kv_dim,
                int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
                const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                base::DeviceType device_type, CudaConfig* config) {
  UNUSED(config);
  CHECK(device_type == base::DeviceType::kDeviceCPU);
  CHECK(!mha_out.is_empty());
  CHECK(!query_tensor.is_empty());
  CHECK(!score_tensor.is_empty());
  CHECK(!key_cache_tensor.is_empty());
  CHECK(!value_cache_tensor.is_empty());
  CHECK(mha_out.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(query_tensor.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(score_tensor.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(key_cache_tensor.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(value_cache_tensor.device_type() == base::DeviceType::kDeviceCPU);
  const base::DataType data_type = mha_out.data_type();
  CHECK(data_type == base::DataType::kDataTypeFp16 ||
        data_type == base::DataType::kDataTypeFp32);
  CHECK(query_tensor.data_type() == data_type);
  CHECK(score_tensor.data_type() == data_type);
  CHECK(key_cache_tensor.data_type() == data_type);
  CHECK(value_cache_tensor.data_type() == data_type);
  CHECK_GE(pos, 0);
  CHECK_GE(layer_index, 0);
  CHECK_LT(pos, seq_len);
  CHECK_GT(seq_len, 0);
  CHECK_GT(head_num, 0);
  CHECK_GT(head_size, 0);
  CHECK_GT(kv_dim, 0);
  CHECK_GT(kv_mul, 0);
  CHECK_EQ(kv_dim % head_size, 0);
  CHECK_LE((head_num + kv_mul - 1) / kv_mul, kv_dim / head_size);
  CHECK_GE(mha_out.size(), static_cast<size_t>(head_num * head_size));
  CHECK_GE(query_tensor.size(), static_cast<size_t>(head_num * head_size));
  CHECK_GE(score_tensor.size(), static_cast<size_t>(head_num * seq_len));
  const size_t cache_size = static_cast<size_t>(layer_index + 1) * seq_len * kv_dim;
  CHECK_GE(key_cache_tensor.size(), cache_size);
  CHECK_GE(value_cache_tensor.size(), cache_size);

  if (data_type == base::DataType::kDataTypeFp16) {
    mha_kernel_impl<__half>(pos, head_num, layer_index, seq_len, kv_dim, kv_mul, head_size,
                            mha_out, query_tensor, score_tensor, key_cache_tensor,
                            value_cache_tensor);
  } else {
    mha_kernel_impl<float>(pos, head_num, layer_index, seq_len, kv_dim, kv_mul, head_size,
                           mha_out, query_tensor, score_tensor, key_cache_tensor,
                           value_cache_tensor);
  }
}
}  // namespace kernel
