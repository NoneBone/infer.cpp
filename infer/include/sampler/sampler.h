#ifndef INFER_SAMPLER_H
#define INFER_SAMPLER_H
#include <cstddef>
#include <cstdint>
#include "base/base.h"

namespace sampler {
class Sampler {
 public:
  explicit Sampler(base::DeviceType device_type) : device_type_(device_type) {}

  virtual size_t sample(const float* logits, size_t size, void* stream = nullptr) = 0;

 protected:
  base::DeviceType device_type_;
};

class ArgmaxSampler : public Sampler {
 public:
  explicit ArgmaxSampler(base::DeviceType device_type) : Sampler(device_type) {}

  size_t sample(const float* logits, size_t size, void* stream) override;
};
}  // namespace sampler
#endif  // INFER_SAMPLER_H
