#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernels/kernels_interface.h"
#include "../source/op/kernels/cpu/rope_kernel.h"
#include "../utils.cuh"
#include "base/buffer.h"
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

TEST(test_rmsnorm_cu, rmsnorm_nostream) {
  if (!has_cuda_device()) GTEST_SKIP() << "CUDA device is unavailable";

  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 15;

  tensor::Tensor in_cpu(base::DataType::kDataTypeFp16, size, true, alloc_cpu);
  tensor::Tensor wei_cpu(base::DataType::kDataTypeFp16, size, true, alloc_cpu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp16, size, true, alloc_cpu);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < size; ++i) {
    in_cpu.index<__half>(i) = __float2half(dist(mt));
    wei_cpu.index<__half>(i) = __float2half(dist(mt));
  }
  tensor::Tensor in_cu = in_cpu.clone();
  tensor::Tensor wei_cu = wei_cpu.clone();
  tensor::Tensor out_cu = out_cpu.clone();
  in_cu.to_cuda(nullptr);
  wei_cu.to_cuda(nullptr);
  out_cu.to_cuda(nullptr);

  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCUDA)(in_cu, wei_cu, out_cu, nullptr);
  out_cu.to_cpu();
  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCPU)(in_cpu, wei_cpu, out_cpu, nullptr);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(out_cu.index<__half>(i), out_cpu.index<__half>(i), __float2half(1e-5f));
  }
}

TEST(test_rmsnorm_cu, abs_error) {
  if (!has_cuda_device()) GTEST_SKIP() << "CUDA device is unavailable";

  constexpr int dim = 257;
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cuda = base::CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor input(base::DataType::kDataTypeFp16, dim, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp16, dim, true, alloc_cpu);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp16, dim, true, alloc_cpu);
  tensor::Tensor output_cuda(base::DataType::kDataTypeFp16, dim, true, alloc_cuda);

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (int i = 0; i < dim; ++i) {
    input.index<__half>(i) = __float2half(dist(rng));
    weight.index<__half>(i) = __float2half(dist(rng));
  }

  auto input_cuda = input.clone();
  auto weight_cuda = weight.clone();
  input_cuda.to_cuda();
  weight_cuda.to_cuda();

  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCPU)(input, weight, output_cpu, nullptr);
  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCUDA)(input_cuda, weight_cuda,
                                                             output_cuda, nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  output_cuda.to_cpu();

  float max_abs = 0.f;
  float max_rel = 0.f;
  for (int i = 0; i < dim; ++i) {
    const float expected = __half2float(output_cpu.index<__half>(i));
    const float actual = __half2float(output_cuda.index<__half>(i));
    const float error = std::fabs(actual - expected);
    max_abs = std::max(max_abs, error);
    max_rel = std::max(max_rel, error / std::max(std::fabs(expected), 1e-6f));
  }

  std::cout << "Metric,MaxAbsoluteError,MaxRelativeError\n"
            << "FP16," << max_abs << "," << max_rel << "\n";
  EXPECT_LE(max_abs, 1e-2f);
  EXPECT_LE(max_rel, 5e-2f);
}

TEST(test_rmsnorm_cu, benchmark) {
  if (!has_cuda_device()) GTEST_SKIP() << "CUDA device is unavailable";

  struct Case {
    const char* name;
    int rows;
    int dim;
  };
  const Case cases[] = {
      {"hidden_norm", 1, 2560},
      {"query_norm", 32, 128},
      {"key_value_norm", 8, 128},
  };
  constexpr int warmup = 10;
  constexpr int iterations = 100;
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cuda = base::CUDADeviceAllocatorFactory::get_instance();
  std::mt19937 rng(5678);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  std::cout << "Time[us],Total Time[ms],Instances,Avg[us],Med[us],Min[us],Max[us],"
               "StdDev[us],GridXYZ,BlockXYZ,Name\n";

  for (const auto& c : cases) {
    tensor::Tensor input(base::DataType::kDataTypeFp16, c.rows, c.dim, true, alloc_cpu);
    tensor::Tensor weight(base::DataType::kDataTypeFp16, c.dim, true, alloc_cpu);
    tensor::Tensor output(base::DataType::kDataTypeFp16, c.rows, c.dim, true, alloc_cuda);
    for (int i = 0; i < input.size(); ++i) {
      input.index<__half>(i) = __float2half(dist(rng));
    }
    for (int i = 0; i < weight.size(); ++i) {
      weight.index<__half>(i) = __float2half(dist(rng));
    }
    input.to_cuda();
    weight.to_cuda();

    for (int i = 0; i < warmup; ++i) {
      kernel::get_rmsnorm_dim_kernel(base::DeviceType::kDeviceCUDA)(
          input, weight, output, c.dim, nullptr);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<double> samples;
    samples.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
      const auto begin = std::chrono::steady_clock::now();
      kernel::get_rmsnorm_dim_kernel(base::DeviceType::kDeviceCUDA)(
          input, weight, output, c.dim, nullptr);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      const auto end = std::chrono::steady_clock::now();
      samples.push_back(
          std::chrono::duration<double, std::micro>(end - begin).count());
    }

    std::sort(samples.begin(), samples.end());
    const double total = std::accumulate(samples.begin(), samples.end(), 0.0);
    const double avg = total / samples.size();
    const double median = samples[samples.size() / 2];
    const double min = samples.front();
    const double max = samples.back();
    double variance = 0.0;
    for (const double value : samples) {
      variance += (value - avg) * (value - avg);
    }
    const double stddev = std::sqrt(variance / samples.size());

    std::cout << std::fixed << std::setprecision(3)
              << avg << "," << total / 1000.0 << "," << samples.size() << ","
              << avg << "," << median << "," << min << "," << max << ","
              << stddev << "," << c.rows << " 1 1,128 1 1,"
              << "\"kernel::rmsnorm_fp16\","
              << c.name << "\n";
  }
}
