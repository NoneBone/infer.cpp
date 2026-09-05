#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "../source/op/kernels/kernels_interface.h"
#include "base/cuda_config.h"
#include "tensor/tensor.h"

namespace {

void fill_tensor(tensor::Tensor& tensor, const std::vector<float>& values) {
  ASSERT_FALSE(tensor.is_empty());
  ASSERT_EQ(static_cast<size_t>(tensor.size()), values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    tensor.index<float>(static_cast<int64_t>(i)) = values[i];
  }
}

std::vector<float> make_expected_output() {
  constexpr float kScale = 0.5f;
  const float score0 = 1.0f * kScale;
  const float score1 = 2.0f * kScale;
  const float max_score = std::max(score0, score1);
  const float exp0 = std::exp(score0 - max_score);
  const float exp1 = std::exp(score1 - max_score);
  const float denom = exp0 + exp1;
  const float w0 = exp0 / denom;
  const float w1 = exp1 / denom;

  const std::vector<float> v0 = {10.f, 20.f, 30.f, 40.f};
  const std::vector<float> v1 = {2.f, 4.f, 6.f, 8.f};
  std::vector<float> expected(4, 0.f);
  for (int i = 0; i < 4; ++i) {
    expected[i] = w0 * v0[i] + w1 * v1[i];
  }
  return expected;
}

}  // namespace

TEST(test_mha_cu, cpu_cuda_match) {
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  kernel::CudaConfig cuda_config;

  constexpr int32_t head_num = 1;
  constexpr int32_t seq_len = 2;
  constexpr int32_t layer_index = 0;
  constexpr int32_t head_size = 4;
  constexpr int32_t kv_dim = 4;
  constexpr int32_t kv_mul = 1;
  constexpr int32_t pos = 1;

  tensor::Tensor query_cpu(base::DataType::kDataTypeFp32, head_num * head_size, true, alloc_cpu);
  tensor::Tensor score_cpu(base::DataType::kDataTypeFp32, head_num * seq_len, true, alloc_cpu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, head_num * head_size, true, alloc_cpu);
  tensor::Tensor key_cache_cpu(base::DataType::kDataTypeFp32, seq_len * kv_dim, true, alloc_cpu);
  tensor::Tensor value_cache_cpu(base::DataType::kDataTypeFp32, seq_len * kv_dim, true, alloc_cpu);

  fill_tensor(query_cpu, {1.f, 2.f, 3.f, 4.f});
  fill_tensor(score_cpu, {0.f, 0.f});
  fill_tensor(out_cpu, {0.f, 0.f, 0.f, 0.f});
  fill_tensor(key_cache_cpu, {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f});
  fill_tensor(value_cache_cpu, {10.f, 20.f, 30.f, 40.f, 2.f, 4.f, 6.f, 8.f});

  tensor::Tensor query_cu = query_cpu.clone();
  tensor::Tensor score_cu = score_cpu.clone();
  tensor::Tensor out_cu = out_cpu.clone();
  tensor::Tensor key_cache_cu = key_cache_cpu.clone();
  tensor::Tensor value_cache_cu = value_cache_cpu.clone();

  query_cu.to_cuda(cuda_config.stream);
  score_cu.to_cuda(cuda_config.stream);
  out_cu.to_cuda(cuda_config.stream);
  key_cache_cu.to_cuda(cuda_config.stream);
  value_cache_cu.to_cuda(cuda_config.stream);

  kernel::get_mha_kernel(base::DeviceType::kDeviceCPU)(
      pos, head_num, layer_index, seq_len, kv_dim, kv_mul, head_size, out_cpu, query_cpu,
      score_cpu, key_cache_cpu, value_cache_cpu, base::DeviceType::kDeviceCPU, nullptr);

  kernel::get_mha_kernel(base::DeviceType::kDeviceCUDA)(
      pos, head_num, layer_index, seq_len, kv_dim, kv_mul, head_size, out_cu, query_cu, score_cu,
      key_cache_cu, value_cache_cu, base::DeviceType::kDeviceCUDA, &cuda_config);

  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  out_cu.to_cpu();

  const auto expected = make_expected_output();
  for (int i = 0; i < head_size; ++i) {
    ASSERT_NEAR(out_cpu.index<float>(i), expected[i], 1e-5f);
    ASSERT_NEAR(out_cu.index<float>(i), expected[i], 1e-5f);
    ASSERT_NEAR(out_cpu.index<float>(i), out_cu.index<float>(i), 1e-5f);
  }
}
