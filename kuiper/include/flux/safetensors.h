#ifndef KUIPER_INCLUDE_FLUX_SAFETENSORS_H_
#define KUIPER_INCLUDE_FLUX_SAFETENSORS_H_

#include <map>
#include <string>
#include <vector>
#include "base/base.h"
#include "tensor/tensor.h"

namespace flux {

struct SafetensorInfo {
  base::DataType data_type = base::DataType::kDataTypeUnknown;
  std::vector<int32_t> shape;
  uint64_t begin = 0;
  uint64_t end = 0;
};

class SafetensorsLoader {
 public:
  static SafetensorsLoader FromFile(const std::string& file_path);
  static SafetensorsLoader FromIndex(const std::string& index_path);

  bool has(const std::string& key) const;
  size_t size() const;
  size_t count_prefix(const std::string& prefix) const;
  const SafetensorInfo& info(const std::string& key) const;

  base::Status load(const std::string& key, tensor::Tensor& tensor,
                    base::DeviceType device = base::DeviceType::kDeviceCPU) const;

 private:
  struct Entry {
    std::string file_path;
    SafetensorInfo info;
  };
  std::map<std::string, Entry> entries_;
};

class FluxWeightMap {
 public:
  void add(const std::string& key, base::DataType data_type,
           const std::vector<int32_t>& shape);
  base::Status validate(const SafetensorsLoader& loader) const;
  base::Status load(const SafetensorsLoader& loader, const std::string& key,
                    tensor::Tensor& tensor,
                    base::DeviceType device = base::DeviceType::kDeviceCPU) const;

 private:
  struct ExpectedTensor {
    base::DataType data_type;
    std::vector<int32_t> shape;
  };
  std::map<std::string, ExpectedTensor> tensors_;
};

}  // namespace flux
#endif  // KUIPER_INCLUDE_FLUX_SAFETENSORS_H_
