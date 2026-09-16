#include "sampler/sampler.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <limits>
#include "../op/kernels/cuda/argmax_kernel.cuh"
namespace sampler {
size_t ArgmaxSampler::sample(const __half* logits, size_t size, void* stream) {
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    size_t next = 0;
    float best = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < size; ++i) {
      float value = __half2float(logits[i]);
      if (value > best) { best = value; next = i; }
    }
    return next;
  } else {
    size_t next = kernel::argmax_kernel_cu(logits, size, stream);
    return next;
  }
}
}  // namespace sampler
