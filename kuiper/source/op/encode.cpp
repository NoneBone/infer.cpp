#include "op/encode.h"
#include <algorithm>
#include <fstream>
#include <limits>
#include <glog/logging.h>
#include "base/unicode.h"
namespace op {

// EncodeLayer::EncodeLayer(
//     base::DeviceType device_type,std::string token_model_path, bool has_bos, bool has_eos,
//     : Layer(device_type, LayerType::kLayerEncode, "Encode"),
//       has_bos_(has_bos),
//       has_eos_(has_eos),
//       spe(std::move(sentence_piece_processor)) {}

std::string SpeEncodeLayer::decode(int32_t token_id) const {
  CHECK(spe != nullptr);
  std::vector<int32_t> token_ids{token_id};
  return this->spe->DecodeIds(token_ids);
}

std::string SpeEncodeLayer::decode(const std::vector<int32_t>& token_ids) const {
  CHECK(spe != nullptr);
  return this->spe->DecodeIds(token_ids);
}

SpeEncodeLayer::SpeEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos)
    : EncodeLayerBase(std::move(token_model_path), has_bos, has_eos) {
  using namespace sentencepiece::util;
  spe = std::make_unique<sentencepiece::SentencePieceProcessor>();
  auto rc = spe->Load(token_model_path_);
  if (rc.code() != StatusCode::kOk) {
    LOG(FATAL)
        << "The token model path is not valid, please check the path and type of token model.";
  }
}

std::vector<int32_t> SpeEncodeLayer::encode(const std::string& sentence) const {
  CHECK(spe != nullptr);
  // sentencepiece
  std::vector<int32_t> input_ids = spe->EncodeAsIds(sentence);
  if (has_bos_) {
    input_ids.insert(input_ids.begin(), spe->bos_id());
  }
  if (has_eos_) {
    input_ids.push_back(spe->eos_id());
  }
  return input_ids;
}

bool SpeEncodeLayer::is_sentence_ending(int32_t token_id) const {
  CHECK(this->spe != nullptr);
  return token_id == this->spe->eos_id();
}

int32_t SpeEncodeLayer::vocab_size() const {
  CHECK(spe != nullptr);
  return spe->GetPieceSize();
}

#if defined(LLAMA3_SUPPORT) || defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
static const std::string PAT_STR =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?:$|[^\S])|\s+)";

namespace {
using json = nlohmann::json;
using TokenMap = ankerl::unordered_dense::map<std::string, int>;

struct LoadedBpeTokens {
  TokenMap encoder;
  TokenMap special_tokens;
  int32_t vocab_size = 0;
};

std::string sibling_path(const std::string& path, const std::string& filename) {
  const auto pos = path.find_last_of("/\\");
  if (pos == std::string::npos) {
    return filename;
  }
  return path.substr(0, pos + 1) + filename;
}

json read_json_file(const std::string& path) {
  std::ifstream f(path);
  CHECK(f.is_open())
      << "The token model path is not valid, please check the path and type of token model: "
      << path;
  try {
    return json::parse(f);
  } catch (json::parse_error&) {
    LOG(FATAL)
        << "The token model path is not valid, please check the path and type of token model: "
        << path;
  }
}

bool json_file_exists(const std::string& path) {
  std::ifstream f(path);
  return f.good();
}

void update_vocab_size(int32_t token_id, int32_t& max_token_id) {
  max_token_id = std::max(max_token_id, token_id);
}

void load_special_tokens_from_json(const json& data, TokenMap& special_tokens,
                                   int32_t& max_token_id) {
  if (data.contains("added_tokens") && data["added_tokens"].is_array()) {
    for (const auto& item : data["added_tokens"]) {
      if (!item.contains("id") || !item.contains("content")) {
        continue;
      }
      const int32_t id = item["id"];
      const std::string content = item["content"];
      special_tokens[content] = id;
      update_vocab_size(id, max_token_id);
    }
  }

  if (data.contains("added_tokens_decoder") && data["added_tokens_decoder"].is_object()) {
    for (const auto& item : data["added_tokens_decoder"].items()) {
      if (!item.value().contains("content")) {
        continue;
      }
      const int32_t id = std::stoi(item.key());
      const std::string content = item.value()["content"];
      special_tokens[content] = id;
      update_vocab_size(id, max_token_id);
    }
  }
}

json load_vocab_json(const std::string& token_model_path, const json& data) {
  if (data.contains("model") && data["model"].contains("vocab") &&
      data["model"]["vocab"].is_object()) {
    return data["model"]["vocab"];
  }

  if (data.is_object() && !data.empty() && data.begin().value().is_number_integer()) {
    return data;
  }

  return read_json_file(sibling_path(token_model_path, "vocab.json"));
}

LoadedBpeTokens load_bpe_tokens(const std::string& token_model_path) {
  const auto data = read_json_file(token_model_path);
  const auto vocab = load_vocab_json(token_model_path, data);

  LoadedBpeTokens tokens;
  int32_t max_token_id = -1;
  load_special_tokens_from_json(data, tokens.special_tokens, max_token_id);
  if (tokens.special_tokens.empty()) {
    const auto tokenizer_config_path = sibling_path(token_model_path, "tokenizer_config.json");
    if (json_file_exists(tokenizer_config_path)) {
      load_special_tokens_from_json(read_json_file(tokenizer_config_path), tokens.special_tokens,
                                    max_token_id);
    }
  }

  CHECK(vocab.is_object()) << "The tokenizer vocab is invalid.";
  for (const auto& v : vocab.items()) {
    const auto cpts = unicode_cpts_from_utf8(v.key());
    std::string key;
    for (const auto cpt : cpts) {
      const auto utf8 = unicode_cpt_to_utf8(cpt);
      key += unicode_utf8_to_byte(utf8);
    }
    const int32_t id = v.value();
    tokens.encoder[key] = id;
    update_vocab_size(id, max_token_id);
  }

  tokens.vocab_size = max_token_id + 1;
  return tokens;
}

int32_t find_special_token_id(const TokenMap& special_tokens, const std::string& token,
                              int32_t default_id = -1) {
  const auto iter = special_tokens.find(token);
  if (iter == special_tokens.end()) {
    return default_id;
  }
  return iter->second;
}
}  // namespace

BpeEncodeLayer::BpeEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos)
    : EncodeLayerBase(std::move(token_model_path), has_bos, has_eos) {
  auto tokens = load_bpe_tokens(token_model_path_);
  bos_id_ = find_special_token_id(tokens.special_tokens, "<|begin_of_text|>");
  eos_id_ = find_special_token_id(tokens.special_tokens, "<|end_of_text|>");
  stop_token1_ = eos_id_;
  stop_token2_ = find_special_token_id(tokens.special_tokens, "<|eot_id|>");

  num_token_ = tokens.vocab_size;
  tiktoken_ =
      std::make_unique<tiktoken::tiktoken>(std::move(tokens.encoder),
                                           std::move(tokens.special_tokens), PAT_STR);
}

std::vector<int32_t> BpeEncodeLayer::encode(const std::string& sentence) const {
  CHECK(this->tiktoken_ != nullptr);
  std::map<std::string, std::string> replacements;
  replacements[" "] = "Ġ";
  std::string s = absl::StrReplaceAll(sentence, replacements);
  auto input_ids = this->tiktoken_->encode(s);

  if (has_bos_) {
    input_ids.insert(input_ids.begin(), bos_id_);
  }
  if (has_eos_) {
    input_ids.push_back(eos_id_);
  }
  return input_ids;
}

std::string BpeEncodeLayer::decode(int32_t token_id) const { return ""; }

std::string BpeEncodeLayer::decode(const std::vector<int32_t>& token_ids) const {
  CHECK(this->tiktoken_ != nullptr);
  auto s = tiktoken_->decode(token_ids);
  std::map<std::string, std::string> reverse_replacements;
  reverse_replacements["Ġ"] = " ";
  const std::string& sentence = absl::StrReplaceAll(s, reverse_replacements);
  return sentence;
}

bool BpeEncodeLayer::is_sentence_ending(int32_t token_id) const {
  if (token_id == stop_token1_ || token_id == stop_token2_) {
    return true;
  } else {
    return false;
  }
}

int32_t BpeEncodeLayer::vocab_size() const {
  CHECK(this->tiktoken_ != nullptr);
  return num_token_;
}

QwenEncodeLayer::QwenEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos)
    : BpeEncodeLayer(std::move(token_model_path), has_bos, has_eos) {
  auto tokens = load_bpe_tokens(token_model_path_);
  bos_id_ = find_special_token_id(tokens.special_tokens, "<|im_start|>");
  eos_id_ = find_special_token_id(tokens.special_tokens, "<|im_end|>");
  stop_token1_ = eos_id_;
  stop_token2_ = find_special_token_id(tokens.special_tokens, "<|endoftext|>");

  CHECK_GE(bos_id_, 0) << "Qwen tokenizer is missing <|im_start|>.";
  CHECK_GE(eos_id_, 0) << "Qwen tokenizer is missing <|im_end|>.";

  num_token_ = tokens.vocab_size;
  tiktoken_ =
      std::make_unique<tiktoken::tiktoken>(std::move(tokens.encoder),
                                           std::move(tokens.special_tokens), PAT_STR);
}

#endif
}  // namespace op
