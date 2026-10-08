#include "cuda_runtime_api.h"
#include "gtest/gtest.h"
#include "glog/logging.h"

#include "base/buffer.h"
#include "tensor/tensor.h"
#include "utils.cuh"
#include "../source/op/kernels/kernels_interface.h"
#include "../source/op/kernels/cuda/matmul_kernel.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <random>
#include <vector>

static bool has_cuda_device() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

TEST(test_matVec_cu, abs_error) {
  if (!has_cuda_device()) GTEST_SKIP() << "CUDA device is unavailable";

  constexpr int output_dim = 257;
  constexpr int input_dim = 513;
  auto cpu = base::CPUDeviceAllocatorFactory::get_instance();
  auto cuda = base::CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor input(base::DataType::kDataTypeFp16, input_dim, true, cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp16, output_dim, input_dim, true, cpu);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp16, output_dim, true, cpu);
  tensor::Tensor output_cuda(base::DataType::kDataTypeFp16, output_dim, true, cuda);

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int i = 0; i < input_dim; ++i) input.index<__half>(i) = __float2half(dist(rng));
  for (int i = 0; i < output_dim * input_dim; ++i)
    weight.index<__half>(i) = __float2half(dist(rng));
  auto input_cuda = input.clone();
  auto weight_cuda = weight.clone();
  input_cuda.to_cuda();
  weight_cuda.to_cuda();

  kernel::get_matmul_kernel(base::DeviceType::kDeviceCPU)(input, weight, output_cpu, 1.f, nullptr);
  kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(input_cuda, weight_cuda, output_cuda, 1.f, nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  output_cuda.to_cpu();

  float max_abs = 0.f;
  float max_rel = 0.f;
  for (int i = 0; i < output_dim; ++i) {
    float expected = __half2float(output_cpu.index<__half>(i));
    float actual = __half2float(output_cuda.index<__half>(i));
    float abs_error = std::fabs(actual - expected);
    max_abs = std::max(max_abs, abs_error);
    max_rel = std::max(max_rel, abs_error / std::max(std::fabs(expected), 1e-6f));
  }
  std::cout << "Metric,MaxAbsoluteError,MaxRelativeError\n"
            << "FP16," << max_abs << "," << max_rel << "\n";
  EXPECT_LE(max_abs, 1e-2f);
  EXPECT_LE(max_rel, 5e-2f);
}

TEST(test_matVec_cu, benchmark) {
  if (!has_cuda_device()) GTEST_SKIP() << "CUDA device is unavailable";

  struct Case { const char* name; int output_dim; int input_dim; };
  const Case cases[] = {
      {"q_proj", 4096, 2560}, {"kv_proj", 512, 2560}, {"o_proj", 2560, 4096},
      {"ffn_up", 9728, 2560}, {"ffn_down", 2560, 9728},
  };
  constexpr int warmup = 10;
  constexpr int iterations = 100;
  auto cpu = base::CPUDeviceAllocatorFactory::get_instance();
  auto cuda = base::CUDADeviceAllocatorFactory::get_instance();
  std::mt19937 rng(5678);
  std::uniform_real_distribution<float> dist(-0.02f, 0.02f);
  std::cout << "Time[us],Total Time[ms],Instances,Avg[us],StdDev[us],GridXYZ,BlockXYZ,Name\n";

  for (const auto& c : cases) {
    tensor::Tensor input(base::DataType::kDataTypeFp16, c.input_dim, true, cpu);
    tensor::Tensor weight(base::DataType::kDataTypeFp16, c.output_dim, c.input_dim, true, cpu);
    tensor::Tensor output(base::DataType::kDataTypeFp16, c.output_dim, true, cuda);
    for (int i = 0; i < input.size(); ++i) input.index<__half>(i) = __float2half(dist(rng));
    for (int i = 0; i < weight.size(); ++i) weight.index<__half>(i) = __float2half(dist(rng));
    input.to_cuda();
    weight.to_cuda();
    for (int i = 0; i < warmup; ++i)
      kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(input, weight, output, 1.f, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<double> samples;
    for (int i = 0; i < iterations; ++i) {
      auto begin = std::chrono::steady_clock::now();
      kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(input, weight, output, 1.f, nullptr);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      auto end = std::chrono::steady_clock::now();
      samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
    }
    double total = std::accumulate(samples.begin(), samples.end(), 0.0);
    double avg = total / samples.size();
    double variance = 0.0;
    for (double value : samples) variance += (value - avg) * (value - avg);
    double stddev = std::sqrt(variance / samples.size());
    int grid = (c.output_dim + 127) / 128;
#if INFER_MATVEC_FP16_BACKEND == 0
    constexpr const char* kernel_name = "matVector_kernel_cu_fp16";
#else
    constexpr const char* kernel_name = "matVector_kernel_cu_fp16_optimized";
#endif
    std::cout << std::fixed << std::setprecision(3)
              << avg << "," << total / 1000.0 << "," << samples.size() << ","
              << avg << "," << stddev << "," << grid << " 1 1,128 1 1,"
              << kernel_name << "," << c.name << "\n";
  }
}
