#include "flux/safetensors.h"

#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include "base/alloc.h"

namespace flux {
namespace {

std::string read_file(const std::string& file_path) {
  std::ifstream file(file_path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("无法打开文件: " + file_path);
  }
  std::ostringstream stream;
  stream << file.rdbuf();
  return stream.str();
}

base::DataType parse_dtype(const std::string& dtype) {
  if (dtype == "BF16") return base::DataType::kDataTypeBf16;
  if (dtype == "F32") return base::DataType::kDataTypeFp32;
  return base::DataType::kDataTypeUnknown;
}

std::vector<int32_t> parse_shape(const std::string& text) {
  std::vector<int32_t> shape;
  const std::regex number(R"((\d+))");
  for (std::sregex_iterator it(text.begin(), text.end(), number), end; it != end; ++it) {
    shape.push_back(std::stoi((*it)[1]));
  }
  return shape;
}

SafetensorInfo parse_info(const std::string& object) {
  const std::regex dtype(R"regex("dtype"\s*:\s*"([^"]+)")regex");
  const std::regex shape(R"regex("shape"\s*:\s*\[([^\]]*)\])regex");
  const std::regex offsets(R"regex("data_offsets"\s*:\s*\[(\d+)\s*,\s*(\d+)\])regex");
  std::smatch match;
  SafetensorInfo info;
  if (!std::regex_search(object, match, dtype)) throw std::runtime_error("safetensors 缺少 dtype");
  info.data_type = parse_dtype(match[1]);
  if (info.data_type == base::DataType::kDataTypeUnknown) {
    throw std::runtime_error("当前仅支持 safetensors BF16/F32");
  }
  if (!std::regex_search(object, match, shape)) throw std::runtime_error("safetensors 缺少 shape");
  info.shape = parse_shape(match[1]);
  if (!std::regex_search(object, match, offsets)) throw std::runtime_error("safetensors 缺少 data_offsets");
  info.begin = std::stoull(match[1]);
  info.end = std::stoull(match[2]);
  size_t elements = 1;
  for (int32_t dim : info.shape) elements *= dim;
  if (info.end < info.begin || info.end - info.begin != elements * base::DataTypeSize(info.data_type)) {
    throw std::runtime_error("safetensors data_offsets 与 tensor shape 不一致");
  }
  return info;
}

size_t matching_brace(const std::string& text, size_t begin) {
  int depth = 0;
  bool quoted = false;
  for (size_t i = begin; i < text.size(); ++i) {
    if (text[i] == '"' && (i == 0 || text[i - 1] != '\\')) quoted = !quoted;
    if (quoted) continue;
    if (text[i] == '{') ++depth;
    if (text[i] == '}' && --depth == 0) return i;
  }
  throw std::runtime_error("JSON 大括号不匹配");
}

std::map<std::string, SafetensorInfo> parse_safetensors_header(const std::string& file_path) {
  std::ifstream file(file_path, std::ios::binary);
  if (!file) throw std::runtime_error("无法打开 safetensors 文件: " + file_path);
  uint64_t header_size = 0;
  file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
  if (!file || header_size == 0 || header_size > (1ULL << 30)) {
    throw std::runtime_error("safetensors header 非法: " + file_path);
  }
  std::string header(header_size, '\0');
  file.read(header.data(), static_cast<std::streamsize>(header_size));
  if (!file) throw std::runtime_error("safetensors header 不完整: " + file_path);

  std::map<std::string, SafetensorInfo> result;
  size_t cursor = 0;
  while ((cursor = header.find('"', cursor)) != std::string::npos) {
    const size_t key_end = header.find('"', cursor + 1);
    const size_t object_begin = header.find('{', key_end);
    if (key_end == std::string::npos || object_begin == std::string::npos) break;
    const std::string key = header.substr(cursor + 1, key_end - cursor - 1);
    const size_t object_end = matching_brace(header, object_begin);
    if (key != "__metadata__") {
      result.emplace(key, parse_info(header.substr(object_begin, object_end - object_begin + 1)));
    }
    cursor = object_end + 1;
  }
  if (result.empty()) throw std::runtime_error("safetensors header 中没有 tensor: " + file_path);
  return result;
}

std::map<std::string, std::string> parse_weight_map(const std::string& text) {
  const size_t marker = text.find("\"weight_map\"");
  const size_t begin = text.find('{', marker);
  if (marker == std::string::npos || begin == std::string::npos) {
    throw std::runtime_error("index.json 缺少 weight_map");
  }
  const size_t end = matching_brace(text, begin);
  std::map<std::string, std::string> result;
  size_t cursor = begin + 1;
  while ((cursor = text.find('"', cursor)) != std::string::npos && cursor < end) {
    const size_t key_end = text.find('"', cursor + 1);
    const size_t value_begin = text.find('"', key_end + 1);
    const size_t value_end = text.find('"', value_begin + 1);
    if (key_end == std::string::npos || value_begin == std::string::npos ||
        value_end == std::string::npos || value_end > end) {
      throw std::runtime_error("index.json 的 weight_map 格式错误");
    }
    result.emplace(text.substr(cursor + 1, key_end - cursor - 1),
                   text.substr(value_begin + 1, value_end - value_begin - 1));
    cursor = value_end + 1;
  }
  if (result.empty()) throw std::runtime_error("index.json 的 weight_map 为空");
  return result;
}

}  // namespace

SafetensorsLoader SafetensorsLoader::FromFile(const std::string& file_path) {
  SafetensorsLoader loader;
  for (const auto& [key, info] : parse_safetensors_header(file_path)) {
    loader.entries_.emplace(key, Entry{file_path, info});
  }
  return loader;
}

SafetensorsLoader SafetensorsLoader::FromIndex(const std::string& index_path) {
  const auto weight_map = parse_weight_map(read_file(index_path));
  const size_t slash = index_path.find_last_of('/');
  const std::string directory = slash == std::string::npos ? "." : index_path.substr(0, slash);
  std::map<std::string, std::map<std::string, SafetensorInfo>> parsed_files;
  SafetensorsLoader loader;
  for (const auto& [key, relative_file] : weight_map) {
    const std::string file_path = directory + "/" + relative_file;
    if (!parsed_files.count(file_path)) parsed_files.emplace(file_path, parse_safetensors_header(file_path));
    const auto tensor_it = parsed_files.at(file_path).find(key);
    if (tensor_it == parsed_files.at(file_path).end()) {
      throw std::runtime_error("index.json 中的 key 不存在于分片: " + key);
    }
    loader.entries_.emplace(key, Entry{file_path, tensor_it->second});
  }
  return loader;
}

bool SafetensorsLoader::has(const std::string& key) const { return entries_.count(key) != 0; }
size_t SafetensorsLoader::size() const { return entries_.size(); }
size_t SafetensorsLoader::count_prefix(const std::string& prefix) const {
  size_t count = 0;
  for (const auto& [key, entry] : entries_) if (key.rfind(prefix, 0) == 0) ++count;
  return count;
}
const SafetensorInfo& SafetensorsLoader::info(const std::string& key) const {
  const auto it = entries_.find(key);
  if (it == entries_.end()) throw std::runtime_error("找不到权重 key: " + key);
  return it->second.info;
}

base::Status SafetensorsLoader::load(const std::string& key, tensor::Tensor& tensor,
                                     base::DeviceType device) const {
  const auto it = entries_.find(key);
  if (it == entries_.end()) return base::error::PathNotValid("找不到权重 key: " + key);
  const Entry& entry = it->second;
  std::ifstream file(entry.file_path, std::ios::binary);
  if (!file) return base::error::PathNotValid("无法打开权重文件: " + entry.file_path);
  uint64_t header_size = 0;
  file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
  const uint64_t offset = sizeof(header_size) + header_size + entry.info.begin;
  const size_t byte_size = entry.info.end - entry.info.begin;
  tensor.reset(entry.info.data_type, entry.info.shape);
  auto cpu_allocator = base::CPUDeviceAllocatorFactory::get_instance();
  if (!tensor.allocate(cpu_allocator)) return base::error::InternalError("CPU 权重内存分配失败");
  file.seekg(static_cast<std::streamoff>(offset));
  file.read(static_cast<char*>(tensor.get_buffer()->ptr()), static_cast<std::streamsize>(byte_size));
  if (!file) return base::error::ModelParseError("权重数据不完整: " + key);
  if (device == base::DeviceType::kDeviceCUDA) tensor.to_cuda();
  if (device != base::DeviceType::kDeviceCPU && device != base::DeviceType::kDeviceCUDA) {
    return base::error::InvalidArgument("不支持的权重目标设备");
  }
  return base::error::Success();
}

void FluxWeightMap::add(const std::string& key, base::DataType data_type,
                        const std::vector<int32_t>& shape) {
  tensors_[key] = ExpectedTensor{data_type, shape};
}

base::Status FluxWeightMap::validate(const SafetensorsLoader& loader) const {
  for (const auto& [key, expected] : tensors_) {
    if (!loader.has(key)) return base::error::PathNotValid("缺少权重 key: " + key);
    const SafetensorInfo& actual = loader.info(key);
    if (actual.data_type != expected.data_type || actual.shape != expected.shape) {
      return base::error::ModelParseError("权重 dtype 或 shape 不匹配: " + key);
    }
  }
  return base::error::Success();
}

base::Status FluxWeightMap::load(const SafetensorsLoader& loader, const std::string& key,
                                 tensor::Tensor& tensor, base::DeviceType device) const {
  const auto expected = tensors_.find(key);
  if (expected == tensors_.end()) return base::error::PathNotValid("WeightMap 未登记 key: " + key);
  const base::Status status = validate(loader);
  if (!status) return status;
  return loader.load(key, tensor, device);
}

}  // namespace flux
