#include "cuda_runtime_api.h"
#include "gtest/gtest.h"

#include "base/buffer.h"
#include "tensor/tensor.h"
#include "../source/op/kernels/cuda/matmul_kernel.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <random>

namespace {

bool has_cuda_device() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

void reference_gemm(const tensor::Tensor& input, const tensor::Tensor& weight,
                    tensor::Tensor& output) {
  const int seq_len = input.get_dim(0);
  const int input_dim = input.get_dim(1);
  const int output_dim = weight.get_dim(0);

  for (int token = 0; token < seq_len; ++token) {
    for (int out = 0; out < output_dim; ++out) {
      float sum = 0.0f;
      for (int k = 0; k < input_dim; ++k) {
        sum += __half2float(input.index<__half>(token * input_dim + k)) *
               __half2float(weight.index<__half>(out * input_dim + k));
      }
      output.index<__half>(token * output_dim + out) = __float2half(sum);
    }
  }
}

}  // namespace

TEST(test_batched_matmul_cu, qkv_projection_matches_reference) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }

  // Qwen3 prefill layout:
  // input  = [seq_len, hidden_size]
  // weight = [output_size, hidden_size]
  // output = [seq_len, output_size]
  constexpr int seq_len = 7;
  constexpr int hidden_size = 13;
  constexpr int q_dim = 11;
  constexpr int kv_dim = 5;

  auto cpu = base::CPUDeviceAllocatorFactory::get_instance();
  auto cuda = base::CUDADeviceAllocatorFactory::get_instance();

  tensor::Tensor input(base::DataType::kDataTypeFp16, seq_len, hidden_size, true, cpu);
  tensor::Tensor q_weight(base::DataType::kDataTypeFp16, q_dim, hidden_size, true, cpu);
  tensor::Tensor k_weight(base::DataType::kDataTypeFp16, kv_dim, hidden_size, true, cpu);
  tensor::Tensor v_weight(base::DataType::kDataTypeFp16, kv_dim, hidden_size, true, cpu);

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  for (int i = 0; i < input.size(); ++i) input.index<__half>(i) = __float2half(dist(rng));
  for (int i = 0; i < q_weight.size(); ++i)
    q_weight.index<__half>(i) = __float2half(dist(rng));
  for (int i = 0; i < k_weight.size(); ++i)
    k_weight.index<__half>(i) = __float2half(dist(rng));
  for (int i = 0; i < v_weight.size(); ++i)
    v_weight.index<__half>(i) = __float2half(dist(rng));

  tensor::Tensor input_cuda = input.clone();
  tensor::Tensor q_weight_cuda = q_weight.clone();
  tensor::Tensor k_weight_cuda = k_weight.clone();
  tensor::Tensor v_weight_cuda = v_weight.clone();
  input_cuda.to_cuda();
  q_weight_cuda.to_cuda();
  k_weight_cuda.to_cuda();
  v_weight_cuda.to_cuda();

  tensor::Tensor q_ref(base::DataType::kDataTypeFp16, seq_len, q_dim, true, cpu);
  tensor::Tensor k_ref(base::DataType::kDataTypeFp16, seq_len, kv_dim, true, cpu);
  tensor::Tensor v_ref(base::DataType::kDataTypeFp16, seq_len, kv_dim, true, cpu);
  reference_gemm(input, q_weight, q_ref);
  reference_gemm(input, k_weight, k_ref);
  reference_gemm(input, v_weight, v_ref);

  tensor::Tensor q_out(base::DataType::kDataTypeFp16, seq_len, q_dim, true, cuda);
  tensor::Tensor k_out(base::DataType::kDataTypeFp16, seq_len, kv_dim, true, cuda);
  tensor::Tensor v_out(base::DataType::kDataTypeFp16, seq_len, kv_dim, true, cuda);

  kernel::batched_matmul_kernel_cu_pure_fp16(input_cuda, q_weight_cuda, q_out, seq_len, 1.f,
                                              nullptr);
  kernel::batched_matmul_kernel_cu_pure_fp16(input_cuda, k_weight_cuda, k_out, seq_len, 1.f,
                                              nullptr);
  kernel::batched_matmul_kernel_cu_pure_fp16(input_cuda, v_weight_cuda, v_out, seq_len, 1.f,
                                              nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  q_out.to_cpu();
  k_out.to_cpu();
  v_out.to_cpu();

  auto check = [](const tensor::Tensor& actual, const tensor::Tensor& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (int i = 0; i < actual.size(); ++i) {
      EXPECT_NEAR(__half2float(actual.index<__half>(i)),
                  __half2float(expected.index<__half>(i)), 1e-2f);
    }
  };
  check(q_out, q_ref);
  check(k_out, k_ref);
  check(v_out, v_ref);
}

TEST(test_batched_matmul_cu, benchmark_qwen3_prefill) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }

  // Shapes follow the Qwen3 workload used by this project.  Each case is:
  // [seq_len, input_dim] x [output_dim, input_dim]^T -> [seq_len, output_dim].
  struct Case {
    const char* name;
    int output_dim;
    int input_dim;
  };
  const Case cases[] = {
      {"q_proj", 4096, 2560},
      {"kv_proj", 512, 2560},
      {"o_proj", 2560, 4096},
      {"ffn_up", 9728, 2560},
      {"ffn_down", 2560, 9728},
  };

  constexpr int seq_len = 128;
  constexpr int warmup = 10;
  constexpr int iterations = 50;
  auto cpu = base::CPUDeviceAllocatorFactory::get_instance();
  auto cuda = base::CUDADeviceAllocatorFactory::get_instance();
  std::mt19937 rng(5678);
  std::uniform_real_distribution<float> dist(-0.02f, 0.02f);

  std::cout << "Time[us],Total Time[ms],Instances,Avg[us],StdDev[us],"
               "GridXYZ,BlockXYZ,Name\n";

  for (const auto& c : cases) {
    tensor::Tensor input(base::DataType::kDataTypeFp16, seq_len, c.input_dim, true, cpu);
    tensor::Tensor weight(base::DataType::kDataTypeFp16, c.output_dim, c.input_dim, true, cpu);
    tensor::Tensor output(base::DataType::kDataTypeFp16, seq_len, c.output_dim, true, cuda);

    for (int i = 0; i < input.size(); ++i) {
      input.index<__half>(i) = __float2half(dist(rng));
    }
    for (int i = 0; i < weight.size(); ++i) {
      weight.index<__half>(i) = __float2half(dist(rng));
    }
    input.to_cuda();
    weight.to_cuda();

    for (int i = 0; i < warmup; ++i) {
      kernel::batched_matmul_kernel_cu_pure_fp16(input, weight, output, seq_len, 1.f, nullptr);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<double> samples;
    samples.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
      auto begin = std::chrono::steady_clock::now();
      kernel::batched_matmul_kernel_cu_pure_fp16(input, weight, output, seq_len, 1.f, nullptr);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      auto end = std::chrono::steady_clock::now();
      samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
    }

    double total = std::accumulate(samples.begin(), samples.end(), 0.0);
    double avg = total / samples.size();
    double variance = 0.0;
    for (double value : samples) {
      variance += (value - avg) * (value - avg);
    }
    double stddev = std::sqrt(variance / samples.size());

    // One block computes one output element: grid=(output_dim, seq_len, 1).
    std::cout << std::fixed << std::setprecision(3)
              << avg << "," << total / 1000.0 << "," << samples.size() << ","
              << avg << "," << stddev << ","
              << c.output_dim << " " << seq_len << " 1,256 1 1,"
              << "batched_gemm_pure_fp16_kernel," << c.name << "\n";
  }
}
