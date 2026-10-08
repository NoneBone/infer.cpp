#define QWEN3_SUPPORT 1
#ifdef QWEN3_SUPPORT
#include "model/qwen3.h"
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <op/matmul.h>
#include <op/mha.h>
#include <op/rmsnorm.h>
#include <op/rope.h>
#include <op/add.h>
#include <op/swiglu.h>
#include <sentencepiece_processor.h>
#include <utility>
#include <cstdlib>
#include "../op/kernels/cpu/rope_kernel.h"
#include "../op/kernels/cuda/rope_kernel.cuh"
#include "../op/kernels/cuda/mha_kernel.cuh"
#include "../op/kernels/kernels_interface.h"
#include "base/tick.h"
namespace model {

void Qwen3Layers::to_cuda(std::shared_ptr<kernel::CudaConfig> config) {
  if (add_layer_) {
    add_layer_->set_cuda_config(config);
    add_layer_->to_cuda();
  }

  if (rope_layer_) {
    rope_layer_->set_cuda_config(config);
    rope_layer_->to_cuda();
  }

  if (swiglu_layer_) {
    swiglu_layer_->set_cuda_config(config);
    swiglu_layer_->to_cuda();
  }

  if (cls_layer_) {
    cls_layer_->set_cuda_config(config);
    cls_layer_->to_cuda();
  }

  if (emb_layer_) {
    emb_layer_->set_cuda_config(config);
    emb_layer_->to_cuda();
  }

  if (mha_layer_) {
    mha_layer_->set_cuda_config(config);
    mha_layer_->to_cuda();
  }

  for (auto& weight_layer : wq_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wk_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wv_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wo_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : w1_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : w2_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : w3_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& rms_norm_layer : rmsnorm_layers_) {
    if (rms_norm_layer) {
      rms_norm_layer->to_cuda();
      rms_norm_layer->set_cuda_config(config);
    }
  }
  if (batched_matmul_helper_layer_) {
    batched_matmul_helper_layer_->set_cuda_config(config);
    batched_matmul_helper_layer_->to_cuda();
  }
}

Qwen3Model::Qwen3Model(base::TokenizerType tokenizer_type, std::string token_path,
                       std::string model_path, bool is_quant_model)
    : Model(tokenizer_type, base::ModelType::kModelTypeLLama2, std::move(token_path),
            std::move(model_path), is_quant_model) {}

base::Status Qwen3Model::init(base::DeviceType device_type) {
  using namespace base;
  if (token_path_.empty()) {
    return error::PathNotValid(token_path_);
  }
  if (device_type == base::DeviceType::kDeviceCPU && is_quant_model_) {
    return error::InternalError("The cpu device do not support int8 quant model.");
  }

  device_type_ = device_type;
  if (device_type == DeviceType::kDeviceCUDA) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0) {
      return error::InternalError("No CUDA device available: " +
                                  std::string(cudaGetErrorString(err)));
    }

    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
      return error::InternalError("cudaSetDevice(0) failed: " +
                                  std::string(cudaGetErrorString(err)));
    }

    cuda_config_ = std::make_shared<kernel::CudaConfig>();
    err = cudaStreamCreate(&cuda_config_->stream);
    if (err != cudaSuccess) {
      return error::InternalError("cudaStreamCreate failed: " +
                                  std::string(cudaGetErrorString(err)));
    }
  }

  base::nvTT.time_push("gen_cost_t");
  Status read_status = gen_model_from_file();
  base::nvTT.time_pop("gen_cost_t");
  base::mP.mem_push("af_gen");
  if (!read_status) {
    return read_status;
  }
  base::nvTT.time_push("pre_alloc");
  init_mem();
  base::nvTT.time_pop("pre_alloc");
  base::mP.mem_push("af_alloc");
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    kernel::sin_cos_cache_calc_cpu(config_->head_size_, config_->seq_len_,
                                   get_buffer(ModelBufferType::kSinCache).ptr<float>(),
                                   get_buffer(ModelBufferType::kCosCache).ptr<float>());
  } else {
    CHECK_NE(cuda_config_, nullptr);
    kernel::sin_cos_cache_calc_cu(config_->head_size_, config_->seq_len_,
                                  get_buffer(ModelBufferType::kSinCache),
                                  get_buffer(ModelBufferType::kCosCache), cuda_config_->stream);
  }

  sampler_ = std::make_unique<sampler::ArgmaxSampler>(device_type_);
  return error::Success();
}

base::Status Qwen3Model::forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                 int& next) const {
  if (input.is_empty()) {
    return base::error::InvalidArgument("The input tensor is empty.");
  }
  if (device_type_ == base::DeviceType::kDeviceCPU && is_quant_model_) {
    return base::error::InternalError("Unsupported int8 quant in the cpu device");
  }

  if (input.dims_size() == 2) {
    if (std::getenv("QWEN3_FORCE_SEQUENTIAL_PREFILL") == nullptr) {
      return prefill(input, input.get_dim(0), pos_tensor.index<int32_t>(0));
    }
  }

  CHECK(pos_tensor.dims_size() == 1);

  // Decode keeps the original single-token GEMV path.  A prompt is currently
  // represented as [seq_len, hidden_dim].  The surrounding attention, RoPE,
  // cache and FFN operators are still single-token operators, so process each
  // row in order while preserving the complete prompt input and its position.
  // This gives correct prefill state and leaves a single place to replace the
  // row loop with a fully batched pipeline later.
  const int32_t token_count = input.dims_size() == 2 ? input.get_dim(0) : 1;
  const int32_t hidden_dim = config_->dim_;
  CHECK_EQ(input.get_dim(input.dims_size() - 1), hidden_dim);
  tensor::Tensor final_input;

  for (int32_t token_idx = 0; token_idx < token_count; ++token_idx) {
    tensor::Tensor token_input;
    if (input.dims_size() == 1) {
      token_input = input;
    } else {
      auto* token_ptr = const_cast<__half*>(input.ptr<__half>(
          static_cast<size_t>(token_idx) * hidden_dim));
      auto token_buffer = std::make_shared<base::Buffer>(
          static_cast<size_t>(hidden_dim) * sizeof(__half), nullptr, token_ptr, true);
      token_input = tensor::Tensor(base::DataType::kDataTypeFp16, hidden_dim);
      CHECK(token_input.assign(token_buffer));
      token_input.set_device_type(device_type_);
    }
    final_input = token_input;

    tensor::Tensor token_pos = pos_tensor;
    token_pos.index<int32_t>(0) = pos_tensor.index<int32_t>(0) + token_idx;
    for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
      attention_rms(layer_idx, token_input);
      attention_qkv(layer_idx, token_pos);
      attention_mha(layer_idx, token_pos);
      feed_forward(layer_idx, token_input);
    }
  }

  // The final token's hidden state is stored in the existing single-token
  // buffers and therefore produces the logits consumed by post_processing.
  cls_logits(final_input);
  return base::error::Success();
}

base::Status Qwen3Model::prefill(const tensor::Tensor& input, int32_t seq_len,
                                  int32_t start_pos, const tensor::Tensor* all_logits) const {
  if (device_type_ != base::DeviceType::kDeviceCUDA) {
    return base::error::InternalError("Prefill GEMM currently requires CUDA");
  }
  if (input.dims_size() != 2 || input.get_dim(0) != seq_len ||
      input.get_dim(1) != config_->dim_ || seq_len <= 0 ||
      start_pos < 0 || start_pos + seq_len > config_->seq_len_) {
    return base::error::InvalidArgument("Invalid prefill input shape or position");
  }

  auto cuda_alloc = base::CUDADeviceAllocatorFactory::get_instance();
  const int32_t dim = config_->dim_;
  const int32_t attn_dim = config_->head_num_ * config_->head_size_;
  const int32_t kv_dim = config_->kv_dim_;
  const int32_t hidden_dim = config_->immediate_dim_;
  const int32_t max_seq = config_->seq_len_;

  tensor::Tensor hidden0(base::DataType::kDataTypeFp16, seq_len, dim, true, cuda_alloc);
  tensor::Tensor hidden1(base::DataType::kDataTypeFp16, seq_len, dim, true, cuda_alloc);
  tensor::Tensor rms_out(base::DataType::kDataTypeFp16, seq_len, dim, true, cuda_alloc);
  tensor::Tensor query(base::DataType::kDataTypeFp16, seq_len, attn_dim, true, cuda_alloc);
  tensor::Tensor key(base::DataType::kDataTypeFp16, seq_len, kv_dim, true, cuda_alloc);
  tensor::Tensor value(base::DataType::kDataTypeFp16, seq_len, kv_dim, true, cuda_alloc);
  tensor::Tensor mha_out(base::DataType::kDataTypeFp16, seq_len, attn_dim, true, cuda_alloc);
  tensor::Tensor wo_out(base::DataType::kDataTypeFp16, seq_len, dim, true, cuda_alloc);
  tensor::Tensor ffn_norm(base::DataType::kDataTypeFp16, seq_len, dim, true, cuda_alloc);
  tensor::Tensor w1_out(base::DataType::kDataTypeFp16, seq_len, hidden_dim, true, cuda_alloc);
  tensor::Tensor w3_out(base::DataType::kDataTypeFp16, seq_len, hidden_dim, true, cuda_alloc);
  tensor::Tensor w2_out(base::DataType::kDataTypeFp16, seq_len, dim, true, cuda_alloc);

  cudaMemcpyAsync(hidden0.ptr<__half>(), input.ptr<__half>(), input.byte_size(),
                  cudaMemcpyDeviceToDevice, cuda_config_->stream);

  for (int32_t layer = 0; layer < config_->layer_num_; ++layer) {
    // Ping-pong hidden states.  Layer 0 reads hidden0 and writes hidden1;
    // every following layer must consume the previous layer's output rather
    // than always reading hidden1.
    const tensor::Tensor& layer_input =
        (layer == 0) ? hidden0 : ((layer % 2 == 1) ? hidden1 : hidden0);
    tensor::Tensor& layer_output = (layer % 2 == 0) ? hidden1 : hidden0;

    if (std::getenv("QWEN3_PREFILL_SINGLE_NORM") != nullptr) {
      auto make_row = [](const tensor::Tensor& tensor, int32_t width, int32_t index) {
        auto buffer = std::make_shared<base::Buffer>(
            static_cast<size_t>(width) * sizeof(__half), nullptr,
            const_cast<__half*>(tensor.ptr<__half>(static_cast<size_t>(index) * width)), true);
        tensor::Tensor result(base::DataType::kDataTypeFp16, width);
        CHECK(result.assign(buffer));
        result.set_device_type(base::DeviceType::kDeviceCUDA);
        return result;
      };
      for (int32_t row = 0; row < seq_len; ++row) {
        STATUS_CHECK(qwen_layers_->rmsnorm_layers_.at(layer)->forward(
            make_row(layer_input, dim, row), make_row(rms_out, dim, row)));
      }
    } else {
      STATUS_CHECK(qwen_layers_->rmsnorm_layers_.at(layer)->forward(layer_input, rms_out));
    }
    if (std::getenv("QWEN3_PREFILL_SINGLE_PROJ") != nullptr) {
      auto make_row = [](const tensor::Tensor& tensor, int32_t width, int32_t index) {
        auto buffer = std::make_shared<base::Buffer>(
            static_cast<size_t>(width) * sizeof(__half), nullptr,
            const_cast<__half*>(tensor.ptr<__half>(static_cast<size_t>(index) * width)), true);
        tensor::Tensor result(base::DataType::kDataTypeFp16, width);
        CHECK(result.assign(buffer));
        result.set_device_type(base::DeviceType::kDeviceCUDA);
        return result;
      };
      auto q = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->wq_layers_.at(layer));
      auto k = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->wk_layers_.at(layer));
      auto v = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->wv_layers_.at(layer));
      CHECK(q != nullptr && k != nullptr && v != nullptr);
      for (int32_t row = 0; row < seq_len; ++row) {
        tensor::Tensor in_row = make_row(rms_out, dim, row);
        tensor::Tensor q_row = make_row(query, attn_dim, row);
        tensor::Tensor k_row = make_row(key, kv_dim, row);
        tensor::Tensor v_row = make_row(value, kv_dim, row);
        q->set_input(0, in_row); q->set_output(0, q_row);
        k->set_input(0, in_row); k->set_output(0, k_row);
        v->set_input(0, in_row); v->set_output(0, v_row);
        STATUS_CHECK(q->forward());
        STATUS_CHECK(k->forward());
        STATUS_CHECK(v->forward());
      }
    } else {
      batched_attention_qkv(layer, rms_out, query, key, value, seq_len, start_pos);
    }

    auto q_norm = qwen_layers_->rmsnorm_layers_.at(2 * config_->layer_num_ + layer + 1);
    auto k_norm = qwen_layers_->rmsnorm_layers_.at(3 * config_->layer_num_ + layer + 1);
    if (std::getenv("QWEN3_PREFILL_SINGLE_QK_NORM") != nullptr) {
      auto make_row = [](const tensor::Tensor& tensor, int32_t width, int32_t index) {
        auto buffer = std::make_shared<base::Buffer>(
            static_cast<size_t>(width) * sizeof(__half), nullptr,
            const_cast<__half*>(tensor.ptr<__half>(static_cast<size_t>(index) * width)), true);
        tensor::Tensor result(base::DataType::kDataTypeFp16, width);
        CHECK(result.assign(buffer));
        result.set_device_type(base::DeviceType::kDeviceCUDA);
        return result;
      };
      for (int32_t row = 0; row < seq_len; ++row) {
        tensor::Tensor q_row = make_row(query, attn_dim, row);
        tensor::Tensor k_row = make_row(key, kv_dim, row);
        q_row.reshape({config_->head_num_, config_->head_size_});
        STATUS_CHECK(q_norm->forward(q_row, q_row));
        k_row.reshape({config_->kv_head_num_, config_->head_size_});
        STATUS_CHECK(k_norm->forward(k_row, k_row));
      }
    } else {
      query.reshape({seq_len * config_->head_num_, config_->head_size_});
      STATUS_CHECK(q_norm->forward(query, query));
      query.reshape({seq_len, attn_dim});
      key.reshape({seq_len * config_->kv_head_num_, config_->head_size_});
      STATUS_CHECK(k_norm->forward(key, key));
      key.reshape({seq_len, kv_dim});
    }

    if (std::getenv("QWEN3_PREFILL_SINGLE_ROPE") != nullptr) {
      auto make_row = [](const tensor::Tensor& tensor, int32_t width, int32_t index) {
        auto buffer = std::make_shared<base::Buffer>(
            static_cast<size_t>(width) * sizeof(__half), nullptr,
            const_cast<__half*>(tensor.ptr<__half>(static_cast<size_t>(index) * width)), true);
        tensor::Tensor result(base::DataType::kDataTypeFp16, width);
        CHECK(result.assign(buffer));
        result.set_device_type(base::DeviceType::kDeviceCUDA);
        return result;
      };
      for (int32_t row = 0; row < seq_len; ++row) {
        tensor::Tensor q_row = make_row(query, attn_dim, row);
        tensor::Tensor k_row = make_row(key, kv_dim, row);
        tensor::Tensor row_pos(base::DataType::kDataTypeInt32, 1, true,
                               base::CPUDeviceAllocatorFactory::get_instance());
        row_pos.index<int32_t>(0) = start_pos + row;
        kernel::rope_kernel_cu(attn_dim, kv_dim, config_->head_size_, q_row, k_row, row_pos,
                               get_buffer(ModelBufferType::kSinCache),
                               get_buffer(ModelBufferType::kCosCache), cuda_config_->stream);
        auto [cache_k, cache_v] = slice_kv_cache(layer, start_pos + row);
        cudaMemcpyAsync(cache_k.ptr<__half>(), k_row.ptr<__half>(),
                        kv_dim * sizeof(__half), cudaMemcpyDeviceToDevice, cuda_config_->stream);
        cudaMemcpyAsync(cache_v.ptr<__half>(),
                        value.ptr<__half>(static_cast<size_t>(row) * kv_dim),
                        kv_dim * sizeof(__half), cudaMemcpyDeviceToDevice, cuda_config_->stream);
      }
    } else {
      kernel::batched_rope_kernel_cu(start_pos, seq_len, attn_dim, kv_dim, config_->head_size_,
                             query, key, get_buffer(ModelBufferType::kSinCache),
                             get_buffer(ModelBufferType::kCosCache), cuda_config_->stream);
      kernel::batched_kv_cache_kernel_cu(layer, start_pos, seq_len, max_seq, kv_dim, key, value,
                                 get_buffer(ModelBufferType::kKeyCache),
                                 get_buffer(ModelBufferType::kValueCache), cuda_config_->stream);
    }

    if (std::getenv("QWEN3_PREFILL_SINGLE_MHA") != nullptr) {
      // Diagnostic path: use the already validated decode MHA kernel for each
      // row while keeping all projections and FFN batched.
      for (int32_t row = 0; row < seq_len; ++row) {
        auto make_row = [](const tensor::Tensor& tensor, int32_t width, int32_t index) {
          auto buffer = std::make_shared<base::Buffer>(
              static_cast<size_t>(width) * sizeof(__half), nullptr,
              const_cast<__half*>(tensor.ptr<__half>(static_cast<size_t>(index) * width)), true);
          tensor::Tensor result(base::DataType::kDataTypeFp16, width);
          CHECK(result.assign(buffer));
          result.set_device_type(base::DeviceType::kDeviceCUDA);
          return result;
        };
        tensor::Tensor q_row = make_row(query, attn_dim, row);
        tensor::Tensor mha_row = make_row(mha_out, attn_dim, row);
        kernel::mha_kernel_cu(start_pos + row, config_->head_num_, layer, max_seq,
                              kv_dim, config_->kv_mul_, config_->head_size_, mha_row, q_row,
                              get_buffer(ModelBufferType::kScoreStorage),
                              get_buffer(ModelBufferType::kKeyCache),
                              get_buffer(ModelBufferType::kValueCache), device_type_,
                              cuda_config_.get());
      }
    } else {
      kernel::batched_mha_kernel_cu(start_pos, seq_len, config_->head_num_, layer, max_seq,
                            attn_dim, kv_dim, config_->kv_mul_, config_->head_size_, mha_out,
                            query, get_buffer(ModelBufferType::kKeyCache),
                            get_buffer(ModelBufferType::kValueCache), cuda_config_.get());
    }
    auto wo = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->wo_layers_.at(layer));
    CHECK(wo != nullptr);
    if (std::getenv("QWEN3_PREFILL_SINGLE_O") != nullptr) {
      for (int32_t row = 0; row < seq_len; ++row) {
        auto make_row = [](const tensor::Tensor& tensor, int32_t width, int32_t index) {
          auto buffer = std::make_shared<base::Buffer>(
              static_cast<size_t>(width) * sizeof(__half), nullptr,
              const_cast<__half*>(tensor.ptr<__half>(static_cast<size_t>(index) * width)), true);
          tensor::Tensor result(base::DataType::kDataTypeFp16, width);
          CHECK(result.assign(buffer));
          result.set_device_type(base::DeviceType::kDeviceCUDA);
          return result;
        };
        tensor::Tensor in_row = make_row(layer_input, dim, row);
        tensor::Tensor mha_row = make_row(mha_out, attn_dim, row);
        tensor::Tensor out_row = make_row(layer_output, dim, row);
        wo->set_input(0, mha_row); wo->set_output(0, out_row);
        STATUS_CHECK(wo->forward());
        kernel::get_add_kernel(device_type_)(in_row, out_row, out_row,
                                             cuda_config_->stream);
      }
    } else {
      STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
          mha_out, wo->get_weight(0), wo_out, seq_len, 1.f));
      kernel::get_add_kernel(device_type_)(layer_input, wo_out, layer_output,
                                           cuda_config_->stream);
    }

    auto w1 = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->w1_layers_.at(layer));
    auto w3 = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->w3_layers_.at(layer));
    auto w2 = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->w2_layers_.at(layer));
    CHECK(w1 != nullptr && w3 != nullptr && w2 != nullptr);
    if (std::getenv("QWEN3_PREFILL_SINGLE_FFN") != nullptr) {
      for (int32_t row = 0; row < seq_len; ++row) {
        auto make_row = [](const tensor::Tensor& tensor, int32_t width, int32_t index) {
          auto buffer = std::make_shared<base::Buffer>(
              static_cast<size_t>(width) * sizeof(__half), nullptr,
              const_cast<__half*>(tensor.ptr<__half>(static_cast<size_t>(index) * width)), true);
          tensor::Tensor result(base::DataType::kDataTypeFp16, width);
          CHECK(result.assign(buffer));
          result.set_device_type(base::DeviceType::kDeviceCUDA);
          return result;
        };
        tensor::Tensor in_row = make_row(layer_output, dim, row);
        tensor::Tensor norm_row = make_row(ffn_norm, dim, row);
        tensor::Tensor w1_row = make_row(w1_out, hidden_dim, row);
        tensor::Tensor w3_row = make_row(w3_out, hidden_dim, row);
        tensor::Tensor w2_row = make_row(w2_out, dim, row);
        STATUS_CHECK(qwen_layers_->rmsnorm_layers_.at(config_->layer_num_ + layer)->forward(
            in_row, norm_row));
        w1->set_input(0, norm_row); w1->set_output(0, w1_row);
        w3->set_input(0, norm_row); w3->set_output(0, w3_row);
        STATUS_CHECK(w1->forward());
        STATUS_CHECK(w3->forward());
        qwen_layers_->swiglu_layer_->set_input(0, w1_row);
        qwen_layers_->swiglu_layer_->set_input(1, w3_row);
        qwen_layers_->swiglu_layer_->set_output(0, w1_row);
        STATUS_CHECK(qwen_layers_->swiglu_layer_->forward());
        w2->set_input(0, w1_row); w2->set_output(0, w2_row);
        STATUS_CHECK(w2->forward());
        kernel::get_add_kernel(device_type_)(in_row, w2_row, in_row,
                                             cuda_config_->stream);
      }
    } else {
      STATUS_CHECK(qwen_layers_->rmsnorm_layers_.at(config_->layer_num_ + layer)->forward(
          layer_output, ffn_norm));
      STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
          ffn_norm, w1->get_weight(0), w1_out, seq_len, 1.f));
      STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
          ffn_norm, w3->get_weight(0), w3_out, seq_len, 1.f));
      kernel::get_swiglu_kernel(device_type_, cuda_config_->stream)(
          w1_out, w3_out, w1_out, cuda_config_->stream);
      STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
          w1_out, w2->get_weight(0), w2_out, seq_len, 1.f));
      kernel::get_add_kernel(device_type_)(layer_output, w2_out, layer_output,
                                           cuda_config_->stream);
    }
  }

  tensor::Tensor& final_hidden = (config_->layer_num_ % 2 == 0) ? hidden0 : hidden1;
  STATUS_CHECK(qwen_layers_->rmsnorm_layers_.at(2 * config_->layer_num_)->forward(
      final_hidden, final_hidden));
  if (all_logits != nullptr) {
    if (all_logits->device_type() != device_type_ ||
        all_logits->data_type() != base::DataType::kDataTypeFp16 ||
        all_logits->dims_size() != 2 || all_logits->get_dim(0) != seq_len ||
        all_logits->get_dim(1) != config_->vocab_size_) {
      return base::error::InvalidArgument("Invalid full prefill logits shape or device");
    }
    auto cls = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->cls_layer_);
    CHECK(cls != nullptr);
    return qwen_layers_->batched_matmul_helper_layer_->forward(
        final_hidden, cls->get_weight(0), *all_logits, seq_len, 1.f);
  }
  auto final_buffer = std::make_shared<base::Buffer>(
      static_cast<size_t>(dim) * sizeof(__half), nullptr,
      const_cast<__half*>(final_hidden.ptr<__half>(static_cast<size_t>(seq_len - 1) * dim)), true);
  tensor::Tensor last(base::DataType::kDataTypeFp16, dim);
  CHECK(last.assign(final_buffer));
  last.set_device_type(device_type_);
  STATUS_CHECK(qwen_layers_->cls_layer_->forward(last, get_buffer(ModelBufferType::kForwardOutput)));
  return base::error::Success();
}

void Qwen3Model::create_nonparam_layers() {
  CHECK(qwen_layers_ != nullptr);
  const int32_t attn_dim = config_->head_num_ * config_->head_size_;
  qwen_layers_->rope_layer_ = std::make_shared<op::RoPELayer>(
      device_type_, attn_dim, config_->kv_dim_, config_->head_size_);

  qwen_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(
      device_type_, 0, config_->kv_mul_, config_->kv_dim_, config_->seq_len_, config_->head_num_,
      config_->head_size_);

  qwen_layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);

  qwen_layers_->swiglu_layer_ =
      std::make_shared<op::SwiGLULayer>(device_type_, config_->immediate_dim_);
  qwen_layers_->batched_matmul_helper_layer_ = std::make_shared<op::BatchedMatmulHelperLayer>(device_type_);
}

void Qwen3Model::create_param_quant_layers() {}

void Qwen3Model::create_param_layers() {
  CHECK(qwen_layers_ != nullptr);

  size_t pos = 0;
  int32_t dim = config_->dim_;
  int32_t kv_dim = config_->kv_dim_;
  int hidden_dim = config_->hidden_dim_;
  auto cpu_device_type = base::DeviceType::kDeviceCPU;
  __half* weight_ptr = (__half*)raw_model_data_->weight(pos);

  // rmsnorm attention attention, ffn, final
  for (int32_t i = 0; i < 2 * config_->layer_num_ + 1; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, hidden_dim);

    rms_norm_layer->set_weight(0, {hidden_dim}, weight_ptr, cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    weight_ptr += hidden_dim;
  }
  pos += (2 * config_->layer_num_ + 1) * hidden_dim;

  // embedding layer
  qwen_layers_->emb_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_, hidden_dim, config_->seq_len_, std::abs(config_->vocab_size_));
  qwen_layers_->emb_layer_->set_weight(0, {std::abs(config_->vocab_size_), hidden_dim},
                                             weight_ptr, cpu_device_type);
  pos += config_->vocab_size_ * hidden_dim;

  // query
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim, false);
    wq->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wq_layers_.push_back(wq);
    pos = pos + hidden_dim * dim;
  }

  // query norm
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, config_->head_size_);
    rms_norm_layer->set_weight(0, {config_->head_size_}, this->raw_model_data_->weight(pos),
                               cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    pos = pos + config_->head_size_;
  }

  // key
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, kv_dim, false);
    wk->set_weight(0, {hidden_dim, kv_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wk_layers_.push_back(wk);
    pos = pos + hidden_dim * kv_dim;
  }

  // key norm
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, config_->head_size_);
    rms_norm_layer->set_weight(0, {config_->head_size_}, this->raw_model_data_->weight(pos),
                               cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    pos = pos + config_->head_size_;
  }

  // value
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, kv_dim, false);
    wv->set_weight(0, {hidden_dim, kv_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wv_layers_.push_back(wv);
    pos += kv_dim * hidden_dim;
  }

  // output
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, false);
    wo->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wo_layers_.push_back(wo);
    pos = pos + dim * hidden_dim;
  }

  // w1 layers
  int32_t immediate_dim = config_->immediate_dim_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, immediate_dim, hidden_dim, false);
    w1->set_weight(0, {immediate_dim, hidden_dim}, this->raw_model_data_->weight(pos),
                   cpu_device_type);
    qwen_layers_->w1_layers_.push_back(w1);
    pos = pos + hidden_dim * immediate_dim;
  }

  // w2 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, immediate_dim, false);
    w2->set_weight(0, {hidden_dim, immediate_dim}, this->raw_model_data_->weight(pos),
                   cpu_device_type);
    qwen_layers_->w2_layers_.push_back(w2);
    pos = pos + immediate_dim * hidden_dim;
  }

  // w3 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, immediate_dim, hidden_dim, false);
    w3->set_weight(0, {immediate_dim, hidden_dim}, this->raw_model_data_->weight(pos),
                   cpu_device_type);
    qwen_layers_->w3_layers_.push_back(w3);
    pos = pos + immediate_dim * hidden_dim;
  }

  auto lm_head = std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_,
                                                   config_->hidden_dim_, false);
  lm_head->set_weight(0, {config_->vocab_size_, config_->hidden_dim_},
                      this->raw_model_data_->weight(pos), cpu_device_type);
  qwen_layers_->cls_layer_ = lm_head;
}

void Qwen3Model::create_param_layers_fp16() {
  CHECK(qwen_layers_ != nullptr);
  LOG(INFO) << "Loading Qwen3 FP16 model weights...";

  size_t pos = 0;
  const int32_t dim = config_->dim_;
  const int32_t kv_dim = config_->kv_dim_;
  const int32_t immediate_dim = config_->immediate_dim_;
  const int32_t attn_dim = config_->head_num_ * config_->head_size_;
  const int32_t vocab_size = config_->vocab_size_;
  const auto cpu = base::DeviceType::kDeviceCPU;

  auto add_norm = [&](int32_t size) {
    auto layer = std::make_shared<op::RmsNormLayer>(device_type_, size);
    layer->set_weight_fp16(0, {size}, raw_model_data_->weight(pos), cpu);
    qwen_layers_->rmsnorm_layers_.push_back(layer);
    pos += size;
  };
  for (int32_t i = 0; i < config_->layer_num_; ++i) add_norm(dim);
  for (int32_t i = 0; i < config_->layer_num_; ++i) add_norm(dim);
  add_norm(dim);

  auto embedding = std::make_shared<op::EmbeddingLayer>(device_type_, dim, config_->seq_len_, vocab_size);
  embedding->set_weight_fp16(0, {vocab_size, dim}, raw_model_data_->weight(pos), cpu);
  qwen_layers_->emb_layer_ = embedding;
  pos += static_cast<size_t>(vocab_size) * dim;

  auto add_matmul = [&](std::vector<std::shared_ptr<op::Layer>>& layers,
                        int32_t out_dim, int32_t in_dim) {
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
      auto layer = std::make_shared<op::MatmulLayer>(device_type_, out_dim, in_dim, false);
      layer->set_weight_fp16(0, {out_dim, in_dim}, raw_model_data_->weight(pos), cpu);
      layers.push_back(layer);
      pos += static_cast<size_t>(out_dim) * in_dim;
    }
  };
  add_matmul(qwen_layers_->wq_layers_, attn_dim, dim);
  add_matmul(qwen_layers_->wk_layers_, kv_dim, dim);
  add_matmul(qwen_layers_->wv_layers_, kv_dim, dim);
  add_matmul(qwen_layers_->wo_layers_, dim, attn_dim);
  add_matmul(qwen_layers_->w1_layers_, immediate_dim, dim);
  add_matmul(qwen_layers_->w2_layers_, dim, immediate_dim);
  add_matmul(qwen_layers_->w3_layers_, immediate_dim, dim);

  auto lm_head = std::make_shared<op::MatmulLayer>(device_type_, vocab_size, dim, false);
  if (config_->is_shared_weight_) {
    const size_t embedding_pos = static_cast<size_t>(2 * config_->layer_num_ + 1) * dim;
    lm_head->set_weight_fp16(0, {vocab_size, dim}, raw_model_data_->weight(embedding_pos), cpu);
  } else {
    lm_head->set_weight_fp16(0, {vocab_size, dim}, raw_model_data_->weight(pos), cpu);
    pos += static_cast<size_t>(vocab_size) * dim;
  }
  qwen_layers_->cls_layer_ = lm_head;

  for (int32_t i = 0; i < config_->layer_num_; ++i) add_norm(config_->head_size_);
  for (int32_t i = 0; i < config_->layer_num_; ++i) add_norm(config_->head_size_);
  LOG(INFO) << "Qwen3 FP16 model loaded successfully. Total FP16 elements: " << pos;
}

void Qwen3Model::init_mem() {
  std::shared_ptr<base::DeviceAllocator> alloc;
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    alloc = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    alloc = base::CUDADeviceAllocatorFactory::get_instance();
  }

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK_NE(cuda_config_, nullptr);
    qwen_layers_->to_cuda(cuda_config_);
  }

  std::shared_ptr<base::DeviceAllocator> alloc_cpu =
      base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::DeviceAllocator> alloc_cu =
      base::CUDADeviceAllocatorFactory::get_instance();
  // 预分配缓冲区，所有中间计算结果都复用这些预分配的缓冲区，避免推理过程中的动态内存分配。
  tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  tensor::Tensor input_embeddings(base::DataType::kDataTypeFp16, 1, config_->dim_, true,
                                  alloc);
  tensor::Tensor sin_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc);
  tensor::Tensor cos_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc);
  // ISSUE: buffermap 设计
  CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
  CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));

  CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
  CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));

  tensor::Tensor rms_output(base::DataType::kDataTypeFp16, config_->dim_, true, alloc);
  tensor::Tensor out_mha(base::DataType::kDataTypeFp16,
                         config_->head_num_ * config_->head_size_, true, alloc);

  CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
  CHECK(insert_buffer(ModelBufferType::kOutputMHA, out_mha));
  CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));
  CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));

  tensor::Tensor w1_output(base::DataType::kDataTypeFp16, config_->immediate_dim_, true, alloc);
  tensor::Tensor w3_output(base::DataType::kDataTypeFp16, config_->immediate_dim_, true, alloc);

  CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));
  CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));

  // kv cache
  tensor::Tensor key_cache(base::DataType::kDataTypeFp16, config_->layer_num_, config_->seq_len_,
                           config_->kv_dim_, true, alloc);
  tensor::Tensor value_cache(base::DataType::kDataTypeFp16, config_->layer_num_, config_->seq_len_,
                             config_->kv_dim_, true, alloc);

  CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
  CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));

  // Wq query output
  tensor::Tensor query(base::DataType::kDataTypeFp16,
                       config_->head_num_ * config_->head_size_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kQuery, query));

  // Pos tensor
  tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));

  // Attention output
  tensor::Tensor attn(base::DataType::kDataTypeFp16, config_->head_num_, config_->seq_len_, true,
                      alloc);
  CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));
  tensor::Tensor attn_output(base::DataType::kDataTypeFp16, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kAttnOutput, attn_output));

  // final forward output
  tensor::Tensor forward_output(base::DataType::kDataTypeFp16, config_->vocab_size_, true, alloc);
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    tensor::Tensor forward_output_cpu(base::DataType::kDataTypeFp32, config_->vocab_size_, true,
                                      alloc_cpu);
    CHECK(insert_buffer(ModelBufferType::kForwardOutputCPU, forward_output_cpu));
  }

  CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
}

base::Status Qwen3Model::create_layers() {
  using namespace base;
  if (!qwen_layers_) {
    qwen_layers_ = std::make_unique<Qwen3Layers>();
  }

  if (!is_quant_model_) {
    create_param_layers_fp16();
  } else {
    return error::FunctionNotImplement("");
  }
  create_nonparam_layers();

  if (!qwen_layers_->emb_layer_) {
    return error::InternalError("Create the embedding layer for the llama model failed!");
  }

  if (qwen_layers_->rmsnorm_layers_.size() != 4 * config_->layer_num_ + 1) {
    // input norm
    return error::InternalError("Create the rmsnorm layers for the llama model failed!");
  }

  if (qwen_layers_->wq_layers_.size() != config_->layer_num_ ||
      qwen_layers_->wk_layers_.size() != config_->layer_num_ ||
      qwen_layers_->wv_layers_.size() != config_->layer_num_ ||
      qwen_layers_->wo_layers_.size() != config_->layer_num_) {
    return error::InternalError(
        "Create the matmul layer in the attention and ffn attention layers for "
        "the llama model "
        "failed.");
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    if (!qwen_layers_->wq_layers_.at(i) || !qwen_layers_->wk_layers_.at(i) ||
        !qwen_layers_->wv_layers_.at(i) || !qwen_layers_->wo_layers_.at(i)) {
      return error::InternalError(
          "Create the matmul layer in the attention and ffn attention layers for "
          "the llama model "
          "failed.");
    }
  }

  if (qwen_layers_->w1_layers_.size() != config_->layer_num_ ||
      qwen_layers_->w2_layers_.size() != config_->layer_num_ ||
      qwen_layers_->w3_layers_.size() != config_->layer_num_) {
    return error::InternalError(
        "Create the matmul layer in the feedforward layers for the llama model "
        "failed.");
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    if (!qwen_layers_->w1_layers_.at(i) || !qwen_layers_->w2_layers_.at(i) ||
        !qwen_layers_->w3_layers_.at(i)) {
      return error::InternalError(
          "Create the matmul layer in the feedforward layers for the llama model "
          "failed.");
    }
  }

  if (!qwen_layers_->rope_layer_) {
    return error::InternalError("Create the rope layer for the llama model failed!");
  }

  if (!qwen_layers_->add_layer_) {
    return error::InternalError("Create the add layer for the llama model failed!");
  }

  if (!qwen_layers_->mha_layer_) {
    return error::InternalError("Create the mha layer for the llama model failed!");
  }

  if (!qwen_layers_->swiglu_layer_) {
    return error::InternalError("Create the SwiGLU layer for the llama model failed!");
  }
  return error::Success();
}

void Qwen3Model::attention_rms(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  // attn rmsnorm
  tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  std::shared_ptr<op::Layer> rmsnorm_layer = qwen_layers_->rmsnorm_layers_.at(layer_idx);
  if (!rmsnorm_layer) {
    LOG(FATAL) << "The attention rmsnorm layer is a null pointer in the llama2 model";
  }
  STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));
}

void Qwen3Model::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(qwen_layers_ != nullptr);
  // kv cache
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  int32_t pos = pos_tensor.index<int32_t>(0);
  // wq wk wv @ input
  auto [key, val] = slice_kv_cache(layer_idx, pos);

  // query
  const auto& query_layer = qwen_layers_->wq_layers_.at(layer_idx);
  CHECK_NE(query_layer, nullptr) << "The query layer in the attention block is null pointer.";

  auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  STATUS_CHECK(query_layer->forward(rmsnorm_output, query));

  // query norm
  auto query_norm = qwen_layers_->rmsnorm_layers_.at(layer_idx + 2 * config_->layer_num_ + 1);
  query.reshape({(int32_t)query.size() / config_->head_size_, config_->head_size_});
  query_norm->forward(query, query);
  query.reshape({(int32_t)query.size()});

  // key
  const auto& key_layer = qwen_layers_->wk_layers_.at(layer_idx);
  CHECK_NE(key_layer, nullptr) << "The key layer in the attention block is null pointer.";
  STATUS_CHECK(key_layer->forward(rmsnorm_output, key));

  // key norm
  auto key_norm = qwen_layers_->rmsnorm_layers_.at(layer_idx + 3 * config_->layer_num_ + 1);
  key.reshape({(int32_t)key.size() / config_->head_size_, config_->head_size_});
  key_norm->forward(key, key);
  key.reshape({(int32_t)key.size()});

  // value
  const auto& value_layer = qwen_layers_->wv_layers_.at(layer_idx);
  CHECK_NE(value_layer, nullptr) << "The value layer in the attention block is null pointer.";
  STATUS_CHECK(value_layer->forward(rmsnorm_output, val));

  // rope
  CHECK_NE(qwen_layers_->rope_layer_, nullptr)
      << "The RoPE layer in the attention block is null pointer.";
  STATUS_CHECK(qwen_layers_->rope_layer_->forward(
      query, key, pos_tensor, get_buffer(ModelBufferType::kSinCache),
      get_buffer(ModelBufferType::kCosCache), tensor::Tensor{}));
}

void Qwen3Model::batched_attention_qkv(int32_t layer_idx, const tensor::Tensor& rms_out,
                                const tensor::Tensor& query_out, const tensor::Tensor& key_out,
                                const tensor::Tensor& value_out,
                                int32_t seq_len, int32_t start_pos) const {
  const auto& query_layer = qwen_layers_->wq_layers_.at(layer_idx);// HERE 关键 gemm
  const auto& key_layer = qwen_layers_->wk_layers_.at(layer_idx);
  const auto& value_layer = qwen_layers_->wv_layers_.at(layer_idx);

  auto query_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(query_layer);
  auto key_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(key_layer);
  auto value_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(value_layer);

  CHECK_NE(query_matmul, nullptr) << "Query layer is not a MatmulLayer";
  CHECK_NE(key_matmul, nullptr) << "Key layer is not a MatmulLayer";
  CHECK_NE(value_matmul, nullptr) << "Value layer is not a MatmulLayer";

  STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
      rms_out, query_matmul->get_weight(0), query_out, seq_len, 1.f));
  STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
      rms_out, key_matmul->get_weight(0), key_out, seq_len, 1.f));
  STATUS_CHECK(qwen_layers_->batched_matmul_helper_layer_->forward(
      rms_out, value_matmul->get_weight(0), value_out, seq_len, 1.f));
}

base::Status Qwen3Model::predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                 bool is_prompt, int& next) const {
  auto status = forward(input, pos_tensor, next);
  if (!status) {
    return status;
  }
  // A batched prompt produces logits for the last prompt row.  Sample them
  // immediately so the caller can switch to the single-token decode path.
  if (input.dims_size() == 2) {
    auto forward_output = get_buffer(ModelBufferType::kForwardOutput);
    next = static_cast<int32_t>(sampler_->sample(
        forward_output.ptr<__half>(), forward_output.size(),
        cuda_config_ ? cuda_config_->stream : nullptr));
  } else {
    next = post_processing(pos_tensor, is_prompt);
  }
  return base::error::Success();
}

void Qwen3Model::attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(qwen_layers_ != nullptr);
  // mha
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  // VAL = [val1,val2,...val t]
  // output @ VAL = 最终的结果
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);

  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
  tensor::Tensor query = get_buffer(ModelBufferType::kQuery);

  const auto& mha_layer = qwen_layers_->mha_layer_;
  CHECK_NE(mha_layer, nullptr) << "The multi head attention layer is null pointer.";
  int pos = pos_tensor.index<int32_t>(0);
  std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_pos(pos);
  std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_layer_idx(layer_idx);
  STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));

  // wo @ attention output
  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  const auto& wo_layer = qwen_layers_->wo_layers_.at(layer_idx);
  CHECK_NE(wo_layer, nullptr) << "The weight output layer is null pointer.";
  STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}

void Qwen3Model::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  // residual add
  CHECK_NE(qwen_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(
      qwen_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input));

  // ffn rmsnorm (post attention layernorm)
  tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  const auto& ffn_rmsnorm = qwen_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  CHECK_NE(ffn_rmsnorm, nullptr)
      << "The final rmsnorm layer in the feedforward block is null pointer";
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));

  // w1
  tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
  const auto& w1_layer = qwen_layers_->w1_layers_.at(layer_idx);
  CHECK_NE(w1_layer, nullptr) << "The w1 layer in the feedforward block is null pointer";
  STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));

  // w3
  tensor::Tensor w3_ouput = get_buffer(ModelBufferType::kW3Output);
  const auto& w3_layer = qwen_layers_->w3_layers_.at(layer_idx);
  CHECK_NE(w3_layer, nullptr) << "The w3 layer in the feedforward block is null pointer";
  STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_ouput));

  // SwiGLU
  CHECK_NE(qwen_layers_->swiglu_layer_, nullptr)
      << "The swiglu layer in the feedforward block is null pointer";
  STATUS_CHECK(qwen_layers_->swiglu_layer_->forward(w1_output, w3_ouput, w1_output));

  // w2
  tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
  const auto& w2_layer = qwen_layers_->w2_layers_.at(layer_idx);
  CHECK_NE(w2_layer, nullptr) << "The w2 layer in the feedforward block is null pointer";
  STATUS_CHECK(w2_layer->forward(w1_output, w2_output));

  // residual add
  CHECK_NE(qwen_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(qwen_layers_->add_layer_->forward(input, w2_output, input));
}

op::EmbeddingOutput Qwen3Model::embedding(const std::vector<int>& tokens) const {
  auto& input_tokens =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kInputTokens));
  auto& input_embeddings =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kInputEmbeddings));
  if (input_tokens.size() != tokens.size()) {
    input_tokens.reshape({static_cast<int32_t>(tokens.size())});
    input_embeddings.reshape({static_cast<int32_t>(tokens.size()), config_->dim_});
  }
  for (int32_t i = 0; i < tokens.size(); ++i) {
    input_tokens.index<int32_t>(i) = tokens.at(i);
  }

  auto input_token_num =
      tensor::Tensor(base::DataType::kDataTypeInt32, static_cast<int32_t>(tokens.size()));
  LOG_IF(FATAL, !qwen_layers_->emb_layer_)
      << "The embedding layer in the llama2 model is null pointer.";
  STATUS_CHECK(
      qwen_layers_->emb_layer_->forward(input_tokens, input_token_num, input_embeddings));

  op::EmbeddingOutput output(input_tokens, input_embeddings, input_token_num);
  return output;
}

tensor::Tensor Model::fill_input(const tensor::Tensor& pos_tensor,
                                 const op::EmbeddingOutput& embedding_output,
                                 bool is_prompt) const {
  auto [input_tokens, input_embeddings, input_token_num] = embedding_output;

  if (is_prompt) {
    return input_embeddings;
  }

  const int32_t pos = pos_tensor.index<int32_t>(0);
  CHECK_EQ(input_embeddings.dims_size(), 2);
  CHECK_GE(pos, 0);
  // Decode embedding() contains only the newly generated token, so its
  // local row is always zero.  `pos` is the global KV-cache position.
  CHECK_EQ(input_embeddings.get_dim(0), 1);
  const int32_t hidden_dim = input_embeddings.get_dim(1);
  const auto* emb_ptr = input_embeddings.ptr<__half>(0);
  std::shared_ptr<base::Buffer> input_emb_buffer = std::make_shared<base::Buffer>(
      static_cast<size_t>(hidden_dim) * sizeof(__half), nullptr,
      const_cast<void*>(static_cast<const void*>(emb_ptr)), true);
  tensor::Tensor input(base::DataType::kDataTypeFp16, hidden_dim);

  input.assign(input_emb_buffer);
  input.set_device_type(device_type_);
  return input;
}

base::Status Model::insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor) {
  if (buffers_.count(buffer_idx) > 0) {
    return base::error::KeyHasExits(std::to_string(int(buffer_idx)) + " has exits in the buffers");
  }
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor is empty for inserting buffer.");
  }
  buffers_.insert({buffer_idx, tensor});
  return base::error::Success();
}

void Qwen3Model::cls_logits(const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  const auto& norm = qwen_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
  CHECK_NE(norm, nullptr);
  STATUS_CHECK(norm->forward(input, input));

  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  CHECK_NE(qwen_layers_->cls_layer_, nullptr);
  STATUS_CHECK(qwen_layers_->cls_layer_->forward(input, forward_output));
}

int32_t Qwen3Model::post_processing(const tensor::Tensor& pos, bool is_prompt) const {
  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  const __half* forward_logits = forward_output.ptr<__half>();

  int32_t next = 0;
  if (is_prompt) {
    next = -1;
  } else {
    next = static_cast<int32_t>(sampler_->sample(forward_logits, forward_output.size(),
                                                 cuda_config_ ? cuda_config_->stream : nullptr));
  }
  return next;
}

}  // namespace model

#endif
