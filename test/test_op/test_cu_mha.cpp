#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "../source/op/kernels/kernels_interface.h"
#include "../source/op/kernels/cuda/mha_kernel.cuh"
#include "base/cuda_config.h"
#include "tensor/tensor.h"

namespace {

bool has_cuda_device() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}


void fill_tensor(tensor::Tensor& tensor, const std::vector<__half>& values) {
  ASSERT_FALSE(tensor.is_empty());
  ASSERT_EQ(static_cast<size_t>(tensor.size()), values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    tensor.index<__half>(static_cast<int64_t>(i)) = values[i];
  }
}

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
  if (!has_cuda_device()) GTEST_SKIP() << "CUDA device is unavailable";

  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  kernel::CudaConfig cuda_config;

  constexpr int32_t head_num = 1;
  constexpr int32_t seq_len = 2;
  constexpr int32_t layer_index = 0;
  constexpr int32_t head_size = 4;
  constexpr int32_t kv_dim = 4;
  constexpr int32_t kv_mul = 1;
  constexpr int32_t pos = 1;

  // CPU 使用 FP32 作为高精度基线，CUDA 使用 FP16 模拟实际推理路径。
  tensor::Tensor query_cpu(base::DataType::kDataTypeFp32, head_num * head_size, true, alloc_cpu);
  tensor::Tensor score_cpu(base::DataType::kDataTypeFp32, head_num * seq_len, true, alloc_cpu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, head_num * head_size, true, alloc_cpu);
  tensor::Tensor key_cache_cpu(base::DataType::kDataTypeFp32, seq_len * kv_dim, true, alloc_cpu);
  tensor::Tensor value_cache_cpu(base::DataType::kDataTypeFp32, seq_len * kv_dim, true, alloc_cpu);

  // 先在 CPU 上构造 FP16 数据，不能直接向 CUDA allocator 返回的设备指针写入。
  tensor::Tensor query_cu_cpu(base::DataType::kDataTypeFp16, head_num * head_size, true,
                              alloc_cpu);
  tensor::Tensor score_cu_cpu(base::DataType::kDataTypeFp16, head_num * seq_len, true, alloc_cpu);
  tensor::Tensor out_cu_cpu(base::DataType::kDataTypeFp16, head_num * head_size, true, alloc_cpu);
  tensor::Tensor key_cache_cu_cpu(base::DataType::kDataTypeFp16, seq_len * kv_dim, true,
                                  alloc_cpu);
  tensor::Tensor value_cache_cu_cpu(base::DataType::kDataTypeFp16, seq_len * kv_dim, true,
                                    alloc_cpu);

  fill_tensor(query_cpu, std::vector<float>{1.f, 2.f, 3.f, 4.f});
  fill_tensor(score_cpu, std::vector<float>{0.f, 0.f});
  fill_tensor(out_cpu, std::vector<float>{0.f, 0.f, 0.f, 0.f});
  fill_tensor(key_cache_cpu, std::vector<float>{1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f});
  fill_tensor(value_cache_cpu,
              std::vector<float>{10.f, 20.f, 30.f, 40.f, 2.f, 4.f, 6.f, 8.f});

  fill_tensor(query_cu_cpu, std::vector<__half>{1.f, 2.f, 3.f, 4.f});
  fill_tensor(score_cu_cpu, std::vector<__half>{0.f, 0.f});
  fill_tensor(out_cu_cpu, std::vector<__half>{0.f, 0.f, 0.f, 0.f});
  fill_tensor(key_cache_cu_cpu,
              std::vector<__half>{1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f});
  fill_tensor(value_cache_cu_cpu,
              std::vector<__half>{10.f, 20.f, 30.f, 40.f, 2.f, 4.f, 6.f, 8.f});

  tensor::Tensor query_cu = query_cu_cpu.clone();
  tensor::Tensor score_cu = score_cu_cpu.clone();
  tensor::Tensor out_cu = out_cu_cpu.clone();
  tensor::Tensor key_cache_cu = key_cache_cu_cpu.clone();
  tensor::Tensor value_cache_cu = value_cache_cu_cpu.clone();

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
    // 输出为 FP16，允许一个以内的 FP16 量化误差。
    ASSERT_NEAR(out_cpu.index<float>(i), expected[i], 1e-5f);
    ASSERT_NEAR(__half2float(out_cu.index<__half>(i)), expected[i], 4e-3f);
  }

  float max_abs_error = 0.f;
  for (int i = 0; i < head_size; ++i) {
    max_abs_error = std::max(
        max_abs_error,
        std::fabs(out_cpu.index<float>(i) - __half2float(out_cu.index<__half>(i))));
  }
  std::cout << "Metric,MaxAbsoluteError\n"
            << "CPU_FP32_vs_CUDA_FP16," << max_abs_error << "\n";
}

TEST(test_mha_cu, batched_prefill_matches_cpu) {
  if (!has_cuda_device()) GTEST_SKIP() << "CUDA device is unavailable";

  constexpr int32_t head_num = 4;
  constexpr int32_t kv_head_num = 2;
  constexpr int32_t head_size = 128;
  constexpr int32_t dim = head_num * head_size;
  constexpr int32_t kv_dim = kv_head_num * head_size;
  constexpr int32_t kv_mul = head_num / kv_head_num;
  constexpr int32_t layer_index = 1;
  constexpr int32_t start_pos = 70;
  constexpr int32_t seq_len = 19;
  constexpr int32_t max_seq_len = 96;

  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  kernel::CudaConfig cuda_config;
  tensor::Tensor query_cpu(base::DataType::kDataTypeFp16, seq_len, dim, true, alloc_cpu);
  tensor::Tensor score_cpu(base::DataType::kDataTypeFp16, seq_len, head_num,
                           max_seq_len, true, alloc_cpu);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp16, seq_len, dim, true, alloc_cpu);
  tensor::Tensor key_cache_cpu(base::DataType::kDataTypeFp16, layer_index + 1,
                               max_seq_len, kv_dim, true, alloc_cpu);
  tensor::Tensor value_cache_cpu(base::DataType::kDataTypeFp16, layer_index + 1,
                                 max_seq_len, kv_dim, true, alloc_cpu);

  std::mt19937 rng(20260918);
  std::uniform_real_distribution<float> dist(-0.2f, 0.2f);
  for (int64_t i = 0; i < query_cpu.size(); ++i) {
    query_cpu.index<__half>(i) = __float2half(dist(rng));
  }
  for (int64_t i = 0; i < key_cache_cpu.size(); ++i) {
    key_cache_cpu.index<__half>(i) = __float2half(dist(rng));
    value_cache_cpu.index<__half>(i) = __float2half(dist(rng));
  }

  std::vector<float> expected(static_cast<size_t>(seq_len) * dim, 0.0f);
  std::vector<float> logits(max_seq_len);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  const size_t layer_base =
      static_cast<size_t>(layer_index) * max_seq_len * kv_dim;
  for (int32_t row = 0; row < seq_len; ++row) {
    const int32_t query_pos = start_pos + row;
    for (int32_t head = 0; head < head_num; ++head) {
      const int32_t kv_head_offset = (head / kv_mul) * head_size;
      float max_logit = -std::numeric_limits<float>::infinity();
      for (int32_t kv_pos = 0; kv_pos <= query_pos; ++kv_pos) {
        float dot = 0.0f;
        for (int32_t d = 0; d < head_size; ++d) {
          const size_t query_index =
              static_cast<size_t>(row) * dim + head * head_size + d;
          const size_t key_index = layer_base +
              static_cast<size_t>(kv_pos) * kv_dim + kv_head_offset + d;
          dot += __half2float(query_cpu.index<__half>(query_index)) *
                 __half2float(key_cache_cpu.index<__half>(key_index));
        }
        logits[kv_pos] = dot * scale;
        max_logit = std::max(max_logit, logits[kv_pos]);
      }
      float sum = 0.0f;
      for (int32_t kv_pos = 0; kv_pos <= query_pos; ++kv_pos) {
        logits[kv_pos] = std::exp(logits[kv_pos] - max_logit);
        sum += logits[kv_pos];
      }
      for (int32_t d = 0; d < head_size; ++d) {
        float value = 0.0f;
        for (int32_t kv_pos = 0; kv_pos <= query_pos; ++kv_pos) {
          const size_t value_index = layer_base +
              static_cast<size_t>(kv_pos) * kv_dim + kv_head_offset + d;
          value += logits[kv_pos] / sum *
                   __half2float(value_cache_cpu.index<__half>(value_index));
        }
        expected[static_cast<size_t>(row) * dim + head * head_size + d] = value;
      }
    }
  }

  tensor::Tensor query = query_cpu.clone();
  tensor::Tensor score = score_cpu.clone();
  tensor::Tensor output = output_cpu.clone();
  tensor::Tensor key_cache = key_cache_cpu.clone();
  tensor::Tensor value_cache = value_cache_cpu.clone();
  query.to_cuda(cuda_config.stream);
  score.to_cuda(cuda_config.stream);
  output.to_cuda(cuda_config.stream);
  key_cache.to_cuda(cuda_config.stream);
  value_cache.to_cuda(cuda_config.stream);

  kernel::batched_mha_kernel_cu(
      start_pos, seq_len, head_num, layer_index, max_seq_len, dim, kv_dim,
      kv_mul, head_size, output, query, score, key_cache, value_cache,
      &cuda_config);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  output.to_cpu();

  float max_abs_error = 0.0f;
  for (int64_t i = 0; i < output.size(); ++i) {
    const float actual = __half2float(output.index<__half>(i));
    ASSERT_TRUE(std::isfinite(actual));
    max_abs_error = std::max(
        max_abs_error, std::fabs(actual - expected[static_cast<size_t>(i)]));
  }
  EXPECT_LT(max_abs_error, 5e-3f);
}

TEST(test_mha_cu, batched_prefill_benchmark) {
  if (!has_cuda_device()) GTEST_SKIP() << "CUDA device is unavailable";

  // Qwen3-4B: 32 个 Query 头、8 个 KV 头、每头 128 维。
  constexpr int32_t head_num = 32;
  constexpr int32_t kv_head_num = 8;
  constexpr int32_t head_size = 128;
  constexpr int32_t dim = head_num * head_size;
  constexpr int32_t kv_dim = kv_head_num * head_size;
  constexpr int32_t kv_mul = head_num / kv_head_num;
  constexpr int32_t layer_index = 0;
  constexpr int32_t warmup = 2;
  constexpr int32_t iterations = 20;
  constexpr int32_t threads_per_block = 128;

  struct Case {
    const char* name;
    int32_t seq_len;
  };
  const Case cases[] = {
      {"prefill_128", 128},
      {"prefill_512", 512},
  };

  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  std::mt19937 rng(5678);
  std::uniform_real_distribution<float> dist(-0.02f, 0.02f);
  kernel::CudaConfig cuda_config;

  std::cout << "Time[us],Total Time[ms],Instances,Avg[us],StdDev[us],GridXYZ,BlockXYZ,Name\n";

  for (const auto& c : cases) {
    tensor::Tensor query_cpu(base::DataType::kDataTypeFp16, c.seq_len, dim, true,
                             alloc_cpu);
    tensor::Tensor score_cpu(base::DataType::kDataTypeFp16, c.seq_len, head_num,
                             c.seq_len, true, alloc_cpu);
    tensor::Tensor output_cpu(base::DataType::kDataTypeFp16, c.seq_len, dim, true,
                              alloc_cpu);
    tensor::Tensor key_cache_cpu(base::DataType::kDataTypeFp16, c.seq_len, kv_dim,
                                 true, alloc_cpu);
    tensor::Tensor value_cache_cpu(base::DataType::kDataTypeFp16, c.seq_len, kv_dim,
                                   true, alloc_cpu);

    for (int64_t i = 0; i < query_cpu.size(); ++i) {
      query_cpu.index<__half>(i) = __float2half(dist(rng));
    }
    for (int64_t i = 0; i < key_cache_cpu.size(); ++i) {
      key_cache_cpu.index<__half>(i) = __float2half(dist(rng));
      value_cache_cpu.index<__half>(i) = __float2half(dist(rng));
    }

    tensor::Tensor query = query_cpu.clone();
    tensor::Tensor score = score_cpu.clone();
    tensor::Tensor output = output_cpu.clone();
    tensor::Tensor key_cache = key_cache_cpu.clone();
    tensor::Tensor value_cache = value_cache_cpu.clone();
    query.to_cuda(cuda_config.stream);
    score.to_cuda(cuda_config.stream);
    output.to_cuda(cuda_config.stream);
    key_cache.to_cuda(cuda_config.stream);
    value_cache.to_cuda(cuda_config.stream);

    for (int32_t w = 0; w < warmup; ++w) {
      kernel::batched_mha_kernel_cu(
          0, c.seq_len, head_num, layer_index, c.seq_len, dim, kv_dim, kv_mul,
          head_size, output, query, score, key_cache, value_cache, &cuda_config);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<double> samples;
    samples.reserve(iterations);
    for (int32_t iter = 0; iter < iterations; ++iter) {
      const auto begin = std::chrono::steady_clock::now();
      kernel::batched_mha_kernel_cu(
          0, c.seq_len, head_num, layer_index, c.seq_len, dim, kv_dim, kv_mul,
          head_size, output, query, score, key_cache, value_cache, &cuda_config);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      const auto end = std::chrono::steady_clock::now();
      samples.push_back(
          std::chrono::duration<double, std::micro>(end - begin).count());
    }

    const double total = std::accumulate(samples.begin(), samples.end(), 0.0);
    const double avg = total / samples.size();
    double variance = 0.0;
    for (const double value : samples) {
      variance += (value - avg) * (value - avg);
    }
    const double stddev = std::sqrt(variance / samples.size());
#if INFER_BATCHED_MHA_BACKEND == 0
    const int32_t grid_x = head_num;
    const int32_t grid_y = c.seq_len;
    constexpr const char* kernel_name = "kernel::batched_mha_fp16";
#else
    constexpr int32_t query_rows_per_block = 16;
    const int32_t grid_x =
        (c.seq_len + query_rows_per_block - 1) / query_rows_per_block;
    const int32_t grid_y = head_num;
    constexpr const char* kernel_name =
        "kernel::batched_mha_prefill_wmma_kernel";
#endif

    std::cout << std::fixed << std::setprecision(3)
              << avg << "," << total / 1000.0 << "," << samples.size() << ","
              << avg << "," << stddev << ","
              << grid_x << " " << grid_y << " 1,"
              << threads_per_block << " 1 1,"
              << "\"" << kernel_name << "\","
              << c.name << "\n";
  }
}

TEST(test_mha_cu, benchmark) {
  if (!has_cuda_device()) GTEST_SKIP() << "CUDA device is unavailable";

  // Qwen3-4B: 32 个 Query 头、8 个 KV 头、每头 128 维。
  constexpr int32_t head_num = 32;
  constexpr int32_t kv_head_num = 8;
  constexpr int32_t head_size = 128;
  constexpr int32_t kv_dim = kv_head_num * head_size;
  constexpr int32_t kv_mul = head_num / kv_head_num;
  constexpr int32_t layer_index = 0;
  constexpr int32_t warmup = 2;
  constexpr int32_t iterations = 20;

  struct Case {
    const char* name;
    int32_t seq_len;
    int32_t prefill_tokens;
    bool prefill;
  };
  const Case cases[] = {
      {"prefill_128", 128, 128, true},
      {"prefill_512", 512, 512, true},
      {"decode_2048", 2048, 1, false},
      {"decode_4096", 4096, 1, false},
  };

  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  std::mt19937 rng(5678);
  std::uniform_real_distribution<float> dist(-0.02f, 0.02f);
  kernel::CudaConfig cuda_config;

  std::cout << "Time[us],Total Time[ms],Instances,Avg[us],StdDev[us],GridXYZ,BlockXYZ,Name\n";

  for (const auto& c : cases) {
    tensor::Tensor query_cpu(base::DataType::kDataTypeFp16, head_num * head_size, true,
                             alloc_cpu);
    tensor::Tensor score_cpu(base::DataType::kDataTypeFp16, head_num * c.seq_len, true,
                             alloc_cpu);
    tensor::Tensor output_cpu(base::DataType::kDataTypeFp16, head_num * head_size, true,
                              alloc_cpu);
    tensor::Tensor key_cache_cpu(base::DataType::kDataTypeFp16, c.seq_len, kv_dim, true,
                                 alloc_cpu);
    tensor::Tensor value_cache_cpu(base::DataType::kDataTypeFp16, c.seq_len, kv_dim, true,
                                   alloc_cpu);

    for (int i = 0; i < query_cpu.size(); ++i) {
      query_cpu.index<__half>(i) = __float2half(dist(rng));
    }
    for (int i = 0; i < score_cpu.size(); ++i) {
      score_cpu.index<__half>(i) = __float2half(0.f);
    }
    for (int i = 0; i < output_cpu.size(); ++i) {
      output_cpu.index<__half>(i) = __float2half(0.f);
    }
    for (int i = 0; i < key_cache_cpu.size(); ++i) {
      key_cache_cpu.index<__half>(i) = __float2half(dist(rng));
      value_cache_cpu.index<__half>(i) = __float2half(dist(rng));
    }

    tensor::Tensor query = query_cpu.clone();
    tensor::Tensor score = score_cpu.clone();
    tensor::Tensor output = output_cpu.clone();
    tensor::Tensor key_cache = key_cache_cpu.clone();
    tensor::Tensor value_cache = value_cache_cpu.clone();
    query.to_cuda(cuda_config.stream);
    score.to_cuda(cuda_config.stream);
    output.to_cuda(cuda_config.stream);
    key_cache.to_cuda(cuda_config.stream);
    value_cache.to_cuda(cuda_config.stream);

    const int32_t call_count = c.prefill ? c.prefill_tokens : 1;
    const int32_t warmup_pos = c.prefill ? c.prefill_tokens - 1 : c.seq_len - 1;
    for (int32_t w = 0; w < warmup; ++w) {
      for (int32_t token = 0; token < call_count; ++token) {
        const int32_t pos = c.prefill ? token : warmup_pos;
        kernel::get_mha_kernel(base::DeviceType::kDeviceCUDA)(
            pos, head_num, layer_index, c.seq_len, kv_dim, kv_mul, head_size, output, query,
            score, key_cache, value_cache, base::DeviceType::kDeviceCUDA, &cuda_config);
      }
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // 每轮代表一个完整场景：Prefill 包含全部 Token，Decode 包含一次调用。
    // 因此每种场景固定产生 100 个 sample。
    std::vector<double> samples;
    samples.reserve(iterations);
    for (int32_t iter = 0; iter < iterations; ++iter) {
      const auto begin = std::chrono::steady_clock::now();
      for (int32_t token = 0; token < call_count; ++token) {
        const int32_t pos = c.prefill ? token : c.seq_len - 1;
        kernel::get_mha_kernel(base::DeviceType::kDeviceCUDA)(
            pos, head_num, layer_index, c.seq_len, kv_dim, kv_mul, head_size, output, query,
            score, key_cache, value_cache, base::DeviceType::kDeviceCUDA, &cuda_config);
      }
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      const auto end = std::chrono::steady_clock::now();
      samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
    }

    const double total = std::accumulate(samples.begin(), samples.end(), 0.0);
    const double avg = total / samples.size();
    double variance = 0.0;
    for (const double value : samples) {
      variance += (value - avg) * (value - avg);
    }
    const double stddev = std::sqrt(variance / samples.size());

    std::cout << std::fixed << std::setprecision(3)
              << avg << "," << total / 1000.0 << "," << samples.size() << ","
              << avg << "," << stddev << ","
              << head_num << " 1 1,256 1 1,"
              << "\"kernel::mha_fp16\","
              << c.name << "\n";
  }
}
