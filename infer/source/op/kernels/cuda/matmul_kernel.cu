#include "matmul_kernel.cuh"
#include <cub/block/block_reduce.cuh>
namespace kernel
{
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp16(const __half* input, const __half* weight, __half* output, int M,
                                      int K) {
  // input:  [K]
  // weight: [M, K]，按 row-major 存储（MatmulLayer 约定为 [output, input]）
  // output: [M]
  // 每个线程负责一个输出元素，使用 FP32 做累加，最后转回 FP16。
  const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (output_idx >= M) {
    return;
  }

  float sum = 0.0f;
  for (int k = 0; k < K; ++k) {
    const float input_value = __half2float(input[k]);
    const float weight_value = __half2float(weight[output_idx * K + k]);
    sum += input_value * weight_value;
  }

  output[output_idx] = __float2half(sum);
}

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, const float scale, const CudaConfig* config) {
  UNUSED(scale);
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t M = weight.get_dim(0);  // output rows
  const int32_t K = weight.get_dim(1);  // input columns
  CHECK_EQ(K, input.get_dim(0));
  CHECK_EQ(output.get_dim(0), M);

  constexpr int threads = 128;
  const int blocks = (M + threads - 1) / threads;
  if (config && config->stream) {
    matmul_kernel_cu_fp16<128, 1><<<blocks, threads, 0, config->stream>>>(
        input.ptr<__half>(), weight.ptr<__half>(), const_cast<__half*>(output.ptr<__half>()), M, K);
  } else {
    matmul_kernel_cu_fp16<128, 1><<<blocks, threads>>>(input.ptr<__half>(), weight.ptr<__half>(),
                                              const_cast<__half*>(output.ptr<__half>()), M, K);
  }
}

/**
 * Batched pure FP16 GEMM kernel for prefill phase
 * FP16 input x FP16 weight -> FP16 output
 * Uses float4 vectorized loads (8 halfs at a time), FP32 accumulation for precision
 */
template <int THREAD_PER_BLOCK>
__global__ void batched_gemm_pure_fp16_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    const int batch_size, const int M, const int K, const float scale
) {
  // One block computes one output element:
  // output[batch, out] = sum_k input[batch, k] * weight[out, k].
  const int out = blockIdx.x;
  const int batch = blockIdx.y;
  const int tid = threadIdx.x;

  if (batch >= batch_size || out >= K) {
    return;
  }

  float sum = 0.0f;
  const int input_offset = batch * M;
  const int weight_offset = out * M;

  for (int k = tid; k < M; k += THREAD_PER_BLOCK) {
    sum += __half2float(input[input_offset + k]) *
           __half2float(weight[weight_offset + k]);
  }

  // Reduce the per-thread partial sums in shared memory.  Accumulating in
  // FP32 avoids introducing FP16 rounding at every multiply-add.
  __shared__ float partial[THREAD_PER_BLOCK];
  partial[tid] = sum;
  __syncthreads();

  for (int stride = THREAD_PER_BLOCK / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[batch * K + out] = __float2half(partial[0] * scale);
  }
}

void batched_matmul_kernel_cu_pure_fp16(const tensor::Tensor& input, const tensor::Tensor& weight,
                                         const tensor::Tensor& output, int32_t batch_size, 
                                         float scale, const CudaConfig* config) {
    CHECK(input.is_empty() == false);
    CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(input.data_type() == base::DataType::kDataTypeFp16);
    CHECK(weight.is_empty() == false && weight.dims_size() == 2);
    CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(weight.data_type() == base::DataType::kDataTypeFp16);
    CHECK(output.data_type() == base::DataType::kDataTypeFp16);
    const int32_t K = weight.get_dim(0);  // output dim
    const int32_t M = weight.get_dim(1);  // input dim

    CHECK(input.dims_size() == 2);
    CHECK(output.dims_size() == 2);
    CHECK_EQ(input.get_dim(0), batch_size);
    CHECK_EQ(input.get_dim(1), M);
    CHECK_EQ(output.get_dim(0), batch_size);
    CHECK_EQ(output.get_dim(1), K);
    
    cudaStream_t stream = config ? config->stream : nullptr;
    
    // custom kernel
    dim3 grid(K, batch_size);
    batched_gemm_pure_fp16_kernel<256><<<grid, 256, 0, stream>>>(
        reinterpret_cast<const half*>(input.ptr<uint16_t>()), 
        reinterpret_cast<const half*>(weight.ptr<uint16_t>()), 
        reinterpret_cast<half*>(const_cast<uint16_t*>(output.ptr<uint16_t>())),
        batch_size, M, K, scale);
  }
} // namespace kernel
