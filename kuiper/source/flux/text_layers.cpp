#include "flux/text_layers.h"
#include <cmath>
#include <vector>
#include "base/alloc.h"
#include "flux/text_kernels.h"
namespace flux {
void linear_cuda(const tensor::Tensor& input, const tensor::Tensor& weight,
                 const tensor::Tensor* bias, tensor::Tensor& output);
void t5_encoder_block_cuda(const tensor::Tensor&, const tensor::Tensor&, const tensor::Tensor&, const tensor::Tensor&, const tensor::Tensor&, const tensor::Tensor&, const tensor::Tensor&, const tensor::Tensor&, const tensor::Tensor&, const tensor::Tensor&, const tensor::Tensor&, int32_t, tensor::Tensor&, const tensor::Tensor*);
void self_attention_cuda(const tensor::Tensor& input, const tensor::Tensor& q_weight,
                         const tensor::Tensor& k_weight, const tensor::Tensor& v_weight,
                         const tensor::Tensor& o_weight, int32_t head_count, bool causal,
                         tensor::Tensor& output, const tensor::Tensor* attention_bias);
namespace {
bool bf16(const tensor::Tensor& t) { return t.data_type() == base::DataType::kDataTypeBf16; }
bool bf16_cpu(const tensor::Tensor& t) {
  return bf16(t) && t.device_type() == base::DeviceType::kDeviceCPU;
}
tensor::Tensor temp(const std::vector<int32_t>& dims) {
  return tensor::Tensor(base::DataType::kDataTypeBf16, dims, true,
                        base::CPUDeviceAllocatorFactory::get_instance());
}
}  // namespace
base::Status embedding(const tensor::Tensor& ids, const tensor::Tensor& token,
                       const tensor::Tensor* position, tensor::Tensor& out) {
  if (ids.data_type() != base::DataType::kDataTypeInt32 || ids.dims_size() != 1 ||
      !bf16_cpu(token) || !bf16_cpu(out) || token.dims_size() != 2 ||
      out.dims() != std::vector<int32_t>({ids.get_dim(0), token.get_dim(1)}))
    return base::error::InvalidArgument("invalid embedding tensors");
  int seq = ids.get_dim(0), hidden = token.get_dim(1), vocab = token.get_dim(0);
  if (position && (!bf16_cpu(*position) || position->dims_size() != 2 ||
                   position->get_dim(1) != hidden || position->get_dim(0) < seq))
    return base::error::InvalidArgument("invalid position embedding");
  for (int s = 0; s < seq; ++s) {
    int id = ids.ptr<int32_t>()[s];
    if (id < 0 || id >= vocab) return base::error::InvalidArgument("token id out of range");
    for (int h = 0; h < hidden; ++h) {
      float v = bf16_to_float(token.ptr<uint16_t>()[id * hidden + h]);
      if (position) v += bf16_to_float(position->ptr<uint16_t>()[s * hidden + h]);
      out.ptr<uint16_t>()[s * hidden + h] = float_to_bf16(v);
    }
  }
  return base::error::Success();
}
base::Status linear(const tensor::Tensor& in, const tensor::Tensor& w, const tensor::Tensor* b,
                    tensor::Tensor& out) {
  if (!bf16(in) || !bf16(w) || !bf16(out) || in.dims_size() != 2 ||
      w.dims_size() != 2 || in.get_dim(1) != w.get_dim(1) ||
      out.dims() != std::vector<int32_t>({in.get_dim(0), w.get_dim(0)}) ||
      in.device_type() != w.device_type() || in.device_type() != out.device_type())
    return base::error::InvalidArgument("invalid linear tensors");
  if (b && (!bf16(*b) || b->dims() != std::vector<int32_t>({w.get_dim(0)}) ||
            b->device_type() != in.device_type()))
    return base::error::InvalidArgument("invalid linear bias");
  if (in.device_type() == base::DeviceType::kDeviceCUDA) {
    linear_cuda(in, w, b, out);
    return base::error::Success();
  }
  if (in.device_type() != base::DeviceType::kDeviceCPU)
    return base::error::InvalidArgument("unsupported linear device");
  int rows = in.get_dim(0), in_dim = in.get_dim(1), out_dim = w.get_dim(0);
  for (int r = 0; r < rows; ++r)
    for (int o = 0; o < out_dim; ++o) {
      float sum = b ? bf16_to_float(b->ptr<uint16_t>()[o]) : 0.f;
      for (int i = 0; i < in_dim; ++i)
        sum += bf16_to_float(in.ptr<uint16_t>()[r * in_dim + i]) *
               bf16_to_float(w.ptr<uint16_t>()[o * in_dim + i]);
      out.ptr<uint16_t>()[r * out_dim + o] = float_to_bf16(sum);
    }
  return base::error::Success();
}
base::Status residual_add(const tensor::Tensor& in, const tensor::Tensor& residual,
                          tensor::Tensor& out) {
  if (!bf16_cpu(in) || !bf16_cpu(residual) || !bf16_cpu(out) || in.dims() != residual.dims() ||
      in.dims() != out.dims())
    return base::error::InvalidArgument("invalid residual tensors");
  for (size_t i = 0; i < in.size(); ++i)
    out.ptr<uint16_t>()[i] = float_to_bf16(bf16_to_float(in.ptr<uint16_t>()[i]) +
                                           bf16_to_float(residual.ptr<uint16_t>()[i]));
  return base::error::Success();
}
base::Status self_attention(const tensor::Tensor& in, const tensor::Tensor& q_w,
                            const tensor::Tensor& k_w, const tensor::Tensor& v_w,
                            const tensor::Tensor& o_w, int32_t heads, bool causal,
                            tensor::Tensor& out, const tensor::Tensor* attention_bias) {
  if (!bf16(in) || !bf16(out) || in.dims_size() != 2 || in.dims() != out.dims() ||
      heads <= 0 || in.get_dim(1) % heads)
    return base::error::InvalidArgument("invalid attention input");
  if (in.device_type() != out.device_type() || in.device_type() != q_w.device_type() ||
      in.device_type() != k_w.device_type() || in.device_type() != v_w.device_type() ||
      in.device_type() != o_w.device_type())
    return base::error::InvalidArgument("attention tensors must share a device");
  if (attention_bias && (!bf16(*attention_bias) || attention_bias->dims() != std::vector<int32_t>({heads, in.get_dim(0), in.get_dim(0)}) || attention_bias->device_type() != in.device_type()))
    return base::error::InvalidArgument("invalid attention bias");
  if (in.device_type() == base::DeviceType::kDeviceCUDA) {
    self_attention_cuda(in, q_w, k_w, v_w, o_w, heads, causal, out, attention_bias);
    return base::error::Success();
  }
  if (in.device_type() != base::DeviceType::kDeviceCPU)
    return base::error::InvalidArgument("unsupported attention device");
  int seq = in.get_dim(0), hidden = in.get_dim(1), head_dim = hidden / heads;
  auto q = temp({seq, hidden}), k = temp({seq, hidden}), v = temp({seq, hidden}),
       context = temp({seq, hidden});
  auto status = linear(in, q_w, nullptr, q);
  if (!status) return status;
  status = linear(in, k_w, nullptr, k);
  if (!status) return status;
  status = linear(in, v_w, nullptr, v);
  if (!status) return status;
  for (int h = 0; h < heads; ++h)
    for (int row = 0; row < seq; ++row) {
      std::vector<float> scores(seq);
      float maxv = -INFINITY;
      for (int col = 0; col < seq; ++col) {
        if (causal && col > row) {
          scores[col] = -INFINITY;
          continue;
        }
        float dot = 0;
        for (int d = 0; d < head_dim; ++d)
          dot += bf16_to_float(q.ptr<uint16_t>()[row * hidden + h * head_dim + d]) *
                 bf16_to_float(k.ptr<uint16_t>()[col * hidden + h * head_dim + d]);
        scores[col] = dot / std::sqrt(float(head_dim)) +
                      (attention_bias ? bf16_to_float(attention_bias->ptr<uint16_t>()[h * seq * seq + row * seq + col]) : 0.f);
        maxv = std::max(maxv, scores[col]);
      }
      float denom = 0;
      for (float& score : scores) {
        score = std::exp(score - maxv);
        denom += score;
      }
      for (int d = 0; d < head_dim; ++d) {
        float sum = 0;
        for (int col = 0; col < seq; ++col)
          sum += scores[col] / denom *
                 bf16_to_float(v.ptr<uint16_t>()[col * hidden + h * head_dim + d]);
        context.ptr<uint16_t>()[row * hidden + h * head_dim + d] = float_to_bf16(sum);
      }
    }
  return linear(context, o_w, nullptr, out);
}
base::Status clip_encoder_block(const tensor::Tensor& input, const tensor::Tensor& norm1_weight,
                                const tensor::Tensor& norm1_bias, const tensor::Tensor& q_weight,
                                const tensor::Tensor& q_bias, const tensor::Tensor& k_weight,
                                const tensor::Tensor& k_bias, const tensor::Tensor& v_weight,
                                const tensor::Tensor& v_bias, const tensor::Tensor& o_weight,
                                const tensor::Tensor& o_bias, const tensor::Tensor& norm2_weight,
                                const tensor::Tensor& norm2_bias, const tensor::Tensor& fc1_weight,
                                const tensor::Tensor& fc1_bias, const tensor::Tensor& fc2_weight,
                                const tensor::Tensor& fc2_bias, int32_t head_count,
                                tensor::Tensor& output) {
  if (!bf16_cpu(input) || !bf16_cpu(output) || input.dims() != output.dims())
    return base::error::InvalidArgument("CLIP block currently requires CPU BF16 tensors");
  auto norm1 = temp(input.dims()), attention = temp(input.dims()), residual1 = temp(input.dims());
  auto norm2 = temp(input.dims()), mlp = temp(input.dims());
  auto status = layer_norm(input, norm1_weight, norm1_bias, norm1, 1e-5f);
  if (!status) return status;
  // CLIP projections include bias; compose them explicitly before causal attention.
  auto q = temp(input.dims()), k = temp(input.dims()), v = temp(input.dims()), context = temp(input.dims());
  status = linear(norm1, q_weight, &q_bias, q); if (!status) return status;
  status = linear(norm1, k_weight, &k_bias, k); if (!status) return status;
  status = linear(norm1, v_weight, &v_bias, v); if (!status) return status;
  const int seq = input.get_dim(0), hidden = input.get_dim(1);
  if (head_count <= 0 || hidden % head_count) return base::error::InvalidArgument("invalid CLIP heads");
  const int head_dim = hidden / head_count;
  for (int h = 0; h < head_count; ++h) for (int row = 0; row < seq; ++row) {
    std::vector<float> score(seq); float maximum = -INFINITY;
    for (int col = 0; col <= row; ++col) { float dot = 0.f;
      for (int d = 0; d < head_dim; ++d) dot += bf16_to_float(q.ptr<uint16_t>()[row*hidden+h*head_dim+d]) * bf16_to_float(k.ptr<uint16_t>()[col*hidden+h*head_dim+d]);
      score[col] = dot / std::sqrt(float(head_dim)); maximum = std::max(maximum, score[col]); }
    float denom = 0.f; for (int col = 0; col <= row; ++col) { score[col] = std::exp(score[col]-maximum); denom += score[col]; }
    for (int d = 0; d < head_dim; ++d) { float value = 0.f; for (int col = 0; col <= row; ++col) value += score[col]/denom * bf16_to_float(v.ptr<uint16_t>()[col*hidden+h*head_dim+d]); context.ptr<uint16_t>()[row*hidden+h*head_dim+d] = float_to_bf16(value); }
  }
  status = linear(context, o_weight, &o_bias, attention); if (!status) return status;
  status = residual_add(input, attention, residual1); if (!status) return status;
  status = layer_norm(residual1, norm2_weight, norm2_bias, norm2, 1e-5f); if (!status) return status;
  status = clip_mlp(norm2, fc1_weight, fc1_bias, fc2_weight, fc2_bias, mlp); if (!status) return status;
  return residual_add(residual1, mlp, output);
}
base::Status clip_mlp(const tensor::Tensor& in, const tensor::Tensor& w1, const tensor::Tensor& b1,
                      const tensor::Tensor& w2, const tensor::Tensor& b2, tensor::Tensor& out) {
  auto hidden = temp({in.get_dim(0), w1.get_dim(0)});
  auto s = linear(in, w1, &b1, hidden);
  if (!s) return s;
  s = quick_gelu(hidden, hidden);
  if (!s) return s;
  return linear(hidden, w2, &b2, out);
}
base::Status t5_encoder_block(const tensor::Tensor& input, const tensor::Tensor& norm1_weight,
                              const tensor::Tensor& q_weight, const tensor::Tensor& k_weight,
                              const tensor::Tensor& v_weight, const tensor::Tensor& o_weight,
                              const tensor::Tensor& relative_attention_bias,
                              const tensor::Tensor& norm2_weight, const tensor::Tensor& wi0_weight,
                              const tensor::Tensor& wi1_weight, const tensor::Tensor& wo_weight,
                              int32_t head_count, tensor::Tensor& output, const tensor::Tensor* attention_mask) {
  if (!bf16(input) || !bf16(output) || input.dims() != output.dims() || head_count <= 0 || input.get_dim(1) % head_count)
    return base::error::InvalidArgument("invalid T5 block tensors");
  if (input.device_type() == base::DeviceType::kDeviceCUDA) {
    t5_encoder_block_cuda(input, norm1_weight, q_weight, k_weight, v_weight, o_weight, relative_attention_bias, norm2_weight, wi0_weight, wi1_weight, wo_weight, head_count, output, attention_mask);
    return base::error::Success();
  }
  if (input.device_type() != base::DeviceType::kDeviceCPU) return base::error::InvalidArgument("unsupported T5 block device");
  const int seq = input.get_dim(0), hidden = input.get_dim(1), head_dim = hidden / head_count;
  auto norm1 = temp(input.dims()), q = temp(input.dims()), k = temp(input.dims()), v = temp(input.dims());
  auto context = temp(input.dims()), attention = temp(input.dims()), residual1 = temp(input.dims());
  auto norm2 = temp(input.dims()), mlp = temp(input.dims());
  auto status = rms_norm(input, norm1_weight, norm1, 1e-6f); if (!status) return status;
  status = linear(norm1, q_weight, nullptr, q); if (!status) return status;
  status = linear(norm1, k_weight, nullptr, k); if (!status) return status;
  status = linear(norm1, v_weight, nullptr, v); if (!status) return status;
  auto bias = temp({head_count, seq, seq});
  status = t5_relative_position_bias(relative_attention_bias, seq, true, bias); if (!status) return status;
  // T5 uses mesh-initialized Q/K and deliberately does not apply 1/sqrt(head_dim) here.
  for (int h = 0; h < head_count; ++h) for (int row = 0; row < seq; ++row) {
    std::vector<float> score(seq); float maximum = -INFINITY;
    for (int col = 0; col < seq; ++col) { float dot = 0.f;
      for (int d = 0; d < head_dim; ++d) dot += bf16_to_float(q.ptr<uint16_t>()[row*hidden+h*head_dim+d]) * bf16_to_float(k.ptr<uint16_t>()[col*hidden+h*head_dim+d]);
      score[col] = dot + bf16_to_float(bias.ptr<uint16_t>()[h*seq*seq+row*seq+col]); maximum = std::max(maximum, score[col]); }
    float denom = 0.f; for (float& item : score) { item = std::exp(item-maximum); denom += item; }
    for (int d = 0; d < head_dim; ++d) { float value = 0.f; for (int col = 0; col < seq; ++col) value += score[col]/denom * bf16_to_float(v.ptr<uint16_t>()[col*hidden+h*head_dim+d]); context.ptr<uint16_t>()[row*hidden+h*head_dim+d] = float_to_bf16(value); }
  }
  status = linear(context, o_weight, nullptr, attention); if (!status) return status;
  status = residual_add(input, attention, residual1); if (!status) return status;
  status = rms_norm(residual1, norm2_weight, norm2, 1e-6f); if (!status) return status;
  status = t5_gated_mlp(norm2, wi0_weight, wi1_weight, wo_weight, mlp); if (!status) return status;
  return residual_add(residual1, mlp, output);
}
base::Status t5_gated_mlp(const tensor::Tensor& in, const tensor::Tensor& w0,
                          const tensor::Tensor& w1, const tensor::Tensor& wo, tensor::Tensor& out) {
  auto a = temp({in.get_dim(0), w0.get_dim(0)}), b = temp({in.get_dim(0), w1.get_dim(0)}),
       g = temp({in.get_dim(0), w0.get_dim(0)});
  auto s = linear(in, w0, nullptr, a);
  if (!s) return s;
  s = linear(in, w1, nullptr, b);
  if (!s) return s;
  s = gated_gelu(b, a, g);
  if (!s) return s;
  return linear(g, wo, nullptr, out);
}
int32_t t5_relative_position_bucket(int32_t relative_position, bool bidirectional,
                                    int32_t num_buckets, int32_t max_distance) {
  int32_t buckets = bidirectional ? num_buckets / 2 : num_buckets;
  int32_t offset = bidirectional && relative_position > 0 ? buckets : 0;
  int32_t distance = bidirectional ? std::abs(relative_position) : std::max(-relative_position, 0);
  int32_t exact = buckets / 2;
  if (distance < exact) return offset + distance;
  float ratio = std::log(float(distance) / exact) / std::log(float(max_distance) / exact);
  return offset + std::min(buckets - 1, exact + int32_t(ratio * (buckets - exact)));
}
base::Status t5_relative_position_bias(const tensor::Tensor& weight, int32_t seq,
                                       bool bidirectional, tensor::Tensor& out) {
  if (!bf16_cpu(weight) || !bf16_cpu(out) || weight.dims_size() != 2 || seq <= 0 ||
      out.dims() != std::vector<int32_t>({weight.get_dim(1), seq, seq}))
    return base::error::InvalidArgument("invalid T5 relative position bias");
  for (int h = 0; h < weight.get_dim(1); ++h) for (int q = 0; q < seq; ++q) for (int k = 0; k < seq; ++k) {
    int b = t5_relative_position_bucket(k - q, bidirectional, weight.get_dim(0));
    out.ptr<uint16_t>()[h * seq * seq + q * seq + k] = weight.ptr<uint16_t>()[b * weight.get_dim(1) + h];
  }
  return base::error::Success();
}
base::Status clip_causal_mask(int32_t seq, tensor::Tensor& out) {
  if (!bf16_cpu(out) || seq <= 0 || out.dims() != std::vector<int32_t>({seq, seq}))
    return base::error::InvalidArgument("invalid CLIP causal mask");
  for (int q = 0; q < seq; ++q) for (int k = 0; k < seq; ++k)
    out.ptr<uint16_t>()[q * seq + k] = float_to_bf16(k > q ? -INFINITY : 0.f);
  return base::error::Success();
}
}  // namespace flux
