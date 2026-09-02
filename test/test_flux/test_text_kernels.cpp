#include <gtest/gtest.h>
#include <flux/text_kernels.h>
#include "base/alloc.h"

namespace {
tensor::Tensor bf16_tensor(const std::vector<int32_t>& dims, const std::vector<float>& values) {
  tensor::Tensor tensor(base::DataType::kDataTypeBf16, dims, true, base::CPUDeviceAllocatorFactory::get_instance());
  for (size_t i = 0; i < values.size(); ++i) tensor.ptr<uint16_t>()[i] = flux::float_to_bf16(values[i]);
  return tensor;
}
float value(const tensor::Tensor& tensor, size_t index) { return flux::bf16_to_float(tensor.ptr<uint16_t>()[index]); }
}

TEST(test_flux_text, layer_norm_and_rms_norm) {
  auto input = bf16_tensor({2, 2}, {1.f, 3.f, 2.f, 4.f});
  auto weight = bf16_tensor({2}, {1.f, 1.f});
  auto bias = bf16_tensor({2}, {0.f, 0.f});
  auto output = bf16_tensor({2, 2}, {0, 0, 0, 0});
  ASSERT_TRUE(flux::layer_norm(input, weight, bias, output, 1e-5f));
  EXPECT_NEAR(value(output, 0), -1.f, 0.02f);
  EXPECT_NEAR(value(output, 1), 1.f, 0.02f);
  ASSERT_TRUE(flux::rms_norm(input, weight, output, 1e-6f));
  EXPECT_NEAR(value(output, 0), 0.447f, 0.02f);
  EXPECT_NEAR(value(output, 1), 1.342f, 0.03f);
}

TEST(test_flux_text, activations_softmax_and_gated_gelu) {
  auto input = bf16_tensor({2, 2}, {-1.f, 0.f, 1.f, 2.f});
  auto output = bf16_tensor({2, 2}, {0, 0, 0, 0});
  ASSERT_TRUE(flux::quick_gelu(input, output));
  EXPECT_NEAR(value(output, 2), 0.846f, 0.02f);
  ASSERT_TRUE(flux::gelu_new(input, output));
  EXPECT_NEAR(value(output, 3), 1.955f, 0.03f);
  ASSERT_TRUE(flux::softmax(input, output));
  EXPECT_NEAR(value(output, 0) + value(output, 1), 1.f, 0.02f);
  auto gate = bf16_tensor({2, 2}, {-1.f, 0.f, 1.f, 2.f});
  ASSERT_TRUE(flux::gated_gelu(input, gate, output));
  EXPECT_NEAR(value(output, 3), 3.91f, 0.08f);
}

TEST(test_flux_text, cuda_matches_cpu) {
  auto input = bf16_tensor({2, 2}, {-1.f, 0.f, 1.f, 2.f});
  auto weight = bf16_tensor({2}, {1.f, 1.f});
  auto bias = bf16_tensor({2}, {0.f, 0.f});
  auto cpu = bf16_tensor({2, 2}, {0, 0, 0, 0});
  ASSERT_TRUE(flux::layer_norm(input, weight, bias, cpu, 1e-5f));

  auto cuda_input = input.clone(); auto cuda_weight = weight.clone();
  auto cuda_bias = bias.clone(); auto cuda_output = cpu.clone();
  cuda_input.to_cuda(); cuda_weight.to_cuda(); cuda_bias.to_cuda(); cuda_output.to_cuda();
  ASSERT_TRUE(flux::layer_norm(cuda_input, cuda_weight, cuda_bias, cuda_output, 1e-5f));
  cuda_output.to_cpu();
  for (size_t i = 0; i < cpu.size(); ++i) EXPECT_NEAR(value(cpu, i), value(cuda_output, i), 0.01f);

  auto cpu_softmax = bf16_tensor({2, 2}, {0, 0, 0, 0});
  ASSERT_TRUE(flux::softmax(input, cpu_softmax));
  cuda_output.to_cuda();
  ASSERT_TRUE(flux::softmax(cuda_input, cuda_output));
  cuda_output.to_cpu();
  for (size_t i = 0; i < cpu_softmax.size(); ++i) EXPECT_NEAR(value(cpu_softmax, i), value(cuda_output, i), 0.01f);
}
