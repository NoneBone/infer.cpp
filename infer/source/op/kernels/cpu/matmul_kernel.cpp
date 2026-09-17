#include "matmul_kernel.h"
#include "../kernels_interface.h"
#include "base/base.h"
#include <cuda_fp16.h>

namespace kernel {
void matmul_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, float scale,
                       const CudaConfig* config) {
  UNUSED(config);
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());
  CHECK(input.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(output.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(input.data_type() == base::DataType::kDataTypeFp16);
  CHECK(weight.data_type() == base::DataType::kDataTypeFp16);
  CHECK(output.data_type() == base::DataType::kDataTypeFp16);
  CHECK_EQ(weight.dims_size(), 2);

  const int32_t input_dim = weight.get_dim(1);
  const int32_t output_dim = weight.get_dim(0);
  int32_t batch = 1;
  if (input.dims_size() == 1) {
    CHECK_EQ(input.get_dim(0), input_dim);
  } else if (input.dims_size() == 2) {
    CHECK_EQ(input.get_dim(1), input_dim);
    batch = input.get_dim(0);
  } else {
    LOG(FATAL) << "The input tensor has a wrong dim size.";
  }
  CHECK_EQ(output.size(), static_cast<size_t>(batch) * output_dim);

  const __half* x = input.ptr<__half>();
  const __half* w = weight.ptr<__half>();
  __half* y = const_cast<__half*>(output.ptr<__half>());
  for (int32_t b = 0; b < batch; ++b) {
    for (int32_t o = 0; o < output_dim; ++o) {
      float sum = 0.0f;
      for (int32_t i = 0; i < input_dim; ++i) {
        sum += __half2float(x[b * input_dim + i]) *
               __half2float(w[o * input_dim + i]);
      }
      y[b * output_dim + o] = __float2half(sum * scale);
    }
  }
}
}  // namespace kernel
