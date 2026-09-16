#include "model/model.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace model {
namespace {
constexpr uint32_t kQwen3Magic = 0x616b3437;  // "ak47"
constexpr size_t kVersionedHeaderSize = 256;
constexpr size_t kLegacyHeaderSize = 7 * sizeof(int32_t);
}

Model::Model(base::TokenizerType tokenizer_type, base::ModelType model_type,
             std::string token_path, std::string model_path, bool is_quant_model)
    : tokenizer_type_(tokenizer_type), model_type_(model_type),
      token_path_(std::move(token_path)), model_path_(std::move(model_path)),
      is_quant_model_(is_quant_model) {}

base::ModelType Model::model_type() const { return model_type_; }
const std::string& Model::token_path() const { return token_path_; }
const std::string& Model::model_path() const { return model_path_; }

tensor::Tensor& Model::get_buffer(ModelBufferType idx) {
  CHECK_GT(buffers_.count(idx), 0) << int(idx); return buffers_.at(idx);
}
const tensor::Tensor& Model::get_buffer(ModelBufferType idx) const {
  CHECK_GT(buffers_.count(idx), 0); return buffers_.at(idx);
}
bool Model::is_sentence_ending(int32_t token) const { CHECK(encode_layer_); return encode_layer_->is_sentence_ending(token); }
std::string Model::decode(int32_t token) const { CHECK(encode_layer_); return encode_layer_->decode(token); }
std::string Model::decode(std::vector<int32_t> tokens) const { CHECK(encode_layer_); return encode_layer_->decode(tokens); }
std::vector<int32_t> Model::encode(const std::string& sentence) const { CHECK(encode_layer_); return encode_layer_->encode(sentence); }

std::pair<tensor::Tensor, tensor::Tensor> Model::slice_kv_cache(int32_t layer, int32_t pos) const {
  size_t offset = (static_cast<size_t>(layer) * config_->seq_len_ + pos) * config_->kv_dim_;
  auto& kc = get_buffer(ModelBufferType::kKeyCache);
  auto& vc = get_buffer(ModelBufferType::kValueCache);
  tensor::Tensor key(base::DataType::kDataTypeFp16, config_->kv_dim_, false, nullptr,
                     const_cast<__half*>(kc.ptr<__half>(offset)));
  tensor::Tensor val(base::DataType::kDataTypeFp16, config_->kv_dim_, false, nullptr,
                     const_cast<__half*>(vc.ptr<__half>(offset)));
  key.set_device_type(device_type_); val.set_device_type(device_type_);
  return {key, val};
}

base::Status Model::create_encode_layer() {
  using namespace base;
  if (tokenizer_type_ == TokenizerType::kEncodeSpe)
    encode_layer_ = std::make_unique<op::SpeEncodeLayer>(token_path_, true, false);
  else
    encode_layer_ = std::make_unique<op::QwenEncodeLayer>(token_path_, false, false);
  if (!encode_layer_) return error::InternalError("Create the encode layer failed.");
  config_->vocab_size_ = encode_layer_->vocab_size();
  if (config_->vocab_size_ <= 0) return error::InternalError("Invalid tokenizer vocabulary size.");
  return error::Success();
}

base::Status Model::generate_model_infos(const ModelConfig& c) const {
  config_->dim_ = c.dim; config_->hidden_dim_ = c.hidden_dim; config_->layer_num_ = c.layer_num;
  config_->head_num_ = c.head_num; config_->kv_head_num_ = c.kv_head_num; config_->seq_len_ = c.seq_len;
  if (c.head_num <= 0 || c.kv_head_num <= 0) return base::error::ModelParseError("Invalid head count.");
  config_->kv_dim_ = c.dim * c.kv_head_num / c.head_num;
  config_->kv_mul_ = c.head_num / c.kv_head_num;
  config_->head_size_ = c.dim / c.head_num;
  config_->immediate_dim_ = c.immediate_dim_;
  config_->is_shared_weight_ = c.vocab_size > 0;
  config_->vocab_size_ = std::abs(c.vocab_size);
  LOG(INFO) << "generate_model_infos: dim_=" << config_->dim_ 
          << ", hidden_dim_=" << config_->hidden_dim_
          << ", immediate_dim_=" << config_->immediate_dim_
          << ", kv_dim_=" << config_->kv_dim_
          << ", head_size_=" << config_->head_size_;
  return base::error::Success();
}

base::Status Model::read_model_file() {
  using namespace base;
  if (model_path_.empty()) return error::PathNotValid("Model path is empty.");
  int fd = open(model_path_.c_str(), O_RDONLY);
  if (fd < 0) return error::PathNotValid("Failed to open model: " + model_path_);
  FILE* file = fdopen(dup(fd), "rb");
  if (!file) { close(fd); return error::PathNotValid("Failed to open model stream."); }

  struct stat sb{};
  if (fstat(fd, &sb) != 0 || sb.st_size <= 0) {
    fclose(file); close(fd); return error::ModelParseError("Invalid model file size.");
  }
  const size_t file_size = static_cast<size_t>(sb.st_size);
  uint32_t magic = 0;
  if (fread(&magic, sizeof(magic), 1, file) != 1) {
    fclose(file); close(fd); return error::ModelParseError("Cannot read model header.");
  }

  bool is_fp16 = false;
  size_t weight_offset = 0;
  ModelConfig c{};
  if (magic == kQwen3Magic) {
    int32_t version = 0;
    uint8_t shared = 0;
    int32_t head_dim = 0;
    if (fread(&version, sizeof(version), 1, file) != 1 || version != 4 ||
        fread(&c, sizeof(int32_t), 7, file) != 7 ||
        fread(&shared, sizeof(shared), 1, file) != 1 ||
        fread(&head_dim, sizeof(head_dim), 1, file) != 1) {
      fclose(file); close(fd); return error::ModelParseError("Invalid Qwen3 FP16 header.");
    }
    c.immediate_dim_ = c.hidden_dim;
    auto status = generate_model_infos(c);
    if (!status) { fclose(file); close(fd); return status; }
    config_->is_shared_weight_ = shared != 0;
    if (head_dim > 0) {
      config_->head_size_ = head_dim;
      config_->kv_dim_ = config_->kv_head_num_ * head_dim;
    }
    is_fp16 = true;
    weight_offset = kVersionedHeaderSize;
  } else {
    if (fseek(file, 0, SEEK_SET) != 0 || fread(&c, sizeof(ModelConfig), 1, file) != 1) {
      fclose(file); close(fd); return error::ModelParseError("Invalid legacy model header.");
    }
    if (is_quant_model_ && fread(&group_size_, sizeof(group_size_), 1, file) != 1) {
      fclose(file); close(fd); return error::ModelParseError("Invalid quantization header.");
    }
    auto status = generate_model_infos(c);
    if (!status) { fclose(file); close(fd); return status; }
    weight_offset = kLegacyHeaderSize + (is_quant_model_ ? sizeof(group_size_) : 0);
  }
  LOG(INFO) << "Config: dim=" << c.dim << ", hidden_dim=" << c.hidden_dim
            << ", layer_num=" << c.layer_num << ", head_num=" << c.head_num
            << ", kv_head_num=" << c.kv_head_num << ", vocab_size=" << c.vocab_size
            << ", seq_len=" << c.seq_len;
  fclose(file);

  if (weight_offset >= file_size) { close(fd); return error::ModelParseError("No weight data in model file."); }
  if (is_fp16) {
    raw_model_data_ = std::make_shared<RawModelDataFp16>();
  } else if (is_quant_model_) {
    raw_model_data_ = std::make_shared<RawModelDataInt8>();
  } else {
    raw_model_data_ = std::make_shared<RawModelDataFp32>();
  }
  raw_model_data_->fd = fd;
  raw_model_data_->file_size = file_size;
  raw_model_data_->data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (raw_model_data_->data == MAP_FAILED || raw_model_data_->data == nullptr)
    return error::ModelParseError("Failed to mmap model file.");
  raw_model_data_->weight_data = static_cast<int8_t*>(raw_model_data_->data) + weight_offset;
  LOG(INFO) << "Loaded model magic=0x" << std::hex << magic << std::dec
            << ", dtype=" << (is_fp16 ? "FP16" : (is_quant_model_ ? "INT8" : "FP32"))
            << ", weight_offset=" << weight_offset;
  return error::Success();
}

base::Status Model::gen_model_from_file() {
  config_ = std::make_unique<TransformerConfig>();
  auto status = create_encode_layer(); if (!status) return status;
  status = read_model_file(); if (!status) return status;
  return create_layers();
}
}  // namespace model
