#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "matmul_kernel.cuh"
#include <cub/block/block_reduce.cuh>

#define BATCHED_MATMUL_FP16_BACKEND 1

#define WARP_SIZE 32
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])

#define LDG32BITS(value) (reinterpret_cast<const half2 *>(&(value))[0])
#define LDG64BITS(value) (reinterpret_cast<const float2 *>(&(value))[0])

#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define LDG128BITS(value) (reinterpret_cast<const float4 *>(&(value))[0])


#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))

#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)

#if ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || (__CUDACC_VER_MAJOR__ > 11)
#define CP_ASYNC_CA(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
#else
#define CP_ASYNC_CA(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
#endif

using namespace nvcuda;
inline __device__ __host__ size_t div_ceil(size_t a, size_t b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

namespace wmma = nvcuda::wmma;
namespace kernel
{
  template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
  __global__ void matVector_kernel_cu_fp16(const __half *input, const __half *weight, __half *output, int M,
                                        int K)
  {
    // input:  [K]
    // weight: [M, K]，按 row-major 存储（MatmulLayer 约定为 [output, input]）
    // output: [M]
    // 每个线程负责一个输出元素，使用 FP32 做累加，最后转回 FP16。
    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= M)
    {
      return;
    }

    float sum = 0.0f;
    for (int k = 0; k < K; ++k)
    {
      const float input_value = __half2float(input[k]);
      const float weight_value = __half2float(weight[output_idx * K + k]);
      sum += input_value * weight_value;
    }

    output[output_idx] = __float2half(sum);
  }

  void matmul_kernel_cu(const tensor::Tensor &input, const tensor::Tensor &weight,
                        const tensor::Tensor &output, const float scale, const CudaConfig *config)
  {
    UNUSED(scale);
    CHECK(input.is_empty() == false && input.dims_size() <= 2);
    CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

    CHECK(weight.is_empty() == false && weight.dims_size() == 2);
    CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
    const int32_t M = weight.get_dim(0); // output rows
    const int32_t K = weight.get_dim(1); // input columns
    CHECK_EQ(K, input.get_dim(0));
    CHECK_EQ(output.get_dim(0), M);

    constexpr int threads = 128;
    const int blocks = (M + threads - 1) / threads;
    if (config && config->stream)
    {
      matVector_kernel_cu_fp16<128, 1><<<blocks, threads, 0, config->stream>>>(
          input.ptr<__half>(), weight.ptr<__half>(), const_cast<__half *>(output.ptr<__half>()), M, K);
    }
    else
    {
      matVector_kernel_cu_fp16<128, 1><<<blocks, threads>>>(input.ptr<__half>(), weight.ptr<__half>(),
                                                         const_cast<__half *>(output.ptr<__half>()), M, K);
    }
  }

#if BATCHED_MATMUL_FP16_BACKEND == 0
  /**
   * One warp computes one 16x16 output tile with Tensor Cores.
   *
   * input:  [batch_size, input_dim]
   * weight: [output_dim, input_dim]
   * output: [batch_size, output_dim] = input * transpose(weight)
   *
   * Shared-memory tiles also provide zero padding for dimensions that are not
   * multiples of 16.  This is needed for short prompts and small unit tests.
   */
  template <int WMMA_M = 16, int WMMA_N = 16, int WMMA_K = 16>
  __global__ void batched_gemm_pure_fp16_kernel(
      const __half *__restrict__ input,
      const __half *__restrict__ weight,
      __half *__restrict__ output,
      const int batch_size, const int input_dim, const int output_dim, const float scale)
  {
    const int tile_row = blockIdx.y * WMMA_M;
    const int tile_col = blockIdx.x * WMMA_N;
    const int lane = threadIdx.x;

    __shared__ __align__(16) __half a_tile[WMMA_M * WMMA_K];
    __shared__ __align__(16) __half b_tile[WMMA_K * WMMA_N];
    __shared__ __align__(16) __half c_tile[WMMA_M * WMMA_N];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half>
        c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    const int num_k_tiles = (input_dim + WMMA_K - 1) / WMMA_K;
    for (int tile_k = 0; tile_k < num_k_tiles; ++tile_k)
    {
      // Materialize A and transpose(weight) as row-major WMMA tiles.
      for (int index = lane; index < WMMA_M * WMMA_K; index += warpSize)
      {
        const int row = index / WMMA_K;
        const int k = index % WMMA_K;
        const int global_row = tile_row + row;
        const int global_k = tile_k * WMMA_K + k;
        a_tile[index] = (global_row < batch_size && global_k < input_dim)
                            ? input[global_row * input_dim + global_k]
                            : __float2half(0.0f);
      }
      for (int index = lane; index < WMMA_K * WMMA_N; index += warpSize)
      {
        const int k = index / WMMA_N;
        const int col = index % WMMA_N;
        const int global_k = tile_k * WMMA_K + k;
        const int global_col = tile_col + col;
        b_tile[index] = (global_k < input_dim && global_col < output_dim)
                            ? weight[global_col * input_dim + global_k]
                            : __float2half(0.0f);
      }
      __syncthreads();

      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half,
                     wmma::row_major>
          a_frag;
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half,
                     wmma::row_major>
          b_frag;
      wmma::load_matrix_sync(a_frag, a_tile, WMMA_K);
      wmma::load_matrix_sync(b_frag, b_tile, WMMA_N);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      __syncthreads();
    }

    wmma::store_matrix_sync(c_tile, c_frag, WMMA_N, wmma::mem_row_major);
    __syncthreads();
    for (int index = lane; index < WMMA_M * WMMA_N; index += warpSize)
    {
      const int row = index / WMMA_N;
      const int col = index % WMMA_N;
      const int global_row = tile_row + row;
      const int global_col = tile_col + col;
      if (global_row < batch_size && global_col < output_dim)
      {
        output[global_row * output_dim + global_col] =
            __float2half(__half2float(c_tile[index]) * scale);
      }
    }
  }
#else

  /**
   * 128x128 block-tiled WMMA GEMM, adapted from
   * hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel.
   *
   * The model stores weight as [output_dim, input_dim].  Keeping the weight
   * tile in that layout and loading matrix_b as col-major makes WMMA observe
   * transpose(weight), without an extra global-memory transpose.
   */
  template <int WMMA_M = 16, int WMMA_N = 16, int WMMA_K = 16,
            int WMMA_TILE_M = 4, int WMMA_TILE_N = 2,
            int WARP_TILE_M = 2, int WARP_TILE_N = 4,
            int OFFSET = 8>
  __global__ void batched_gemm_wmma_async_kernel(
      const half *__restrict__ input,
      const half *__restrict__ weight,
      half *__restrict__ output,
      int batch_size, int output_dim, int input_dim, float scale)
  {
    constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 128
    constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 128
    constexpr int BK = WMMA_K;                              // 16

    union SharedStorage
    {
      struct
      {
        half a[2][BM][BK + OFFSET];
        // Stored as [N, K]. WMMA reads it as col-major [K, N].
        half weight[2][BN][BK + OFFSET];
      } operands;
      half c[BM][BN];
    };
    __shared__ __align__(16) SharedStorage shared;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_m = warp_id / WMMA_TILE_N;
    const int warp_n = warp_id % WMMA_TILE_N;

    const int load_row = tid / 2;             // 0..127
    const int load_k = (tid & 1) * 8;         // 0 or 8
    const int global_a_row = blockIdx.y * BM + load_row;
    const int global_w_row = blockIdx.x * BN + load_row;
    const int num_k_tiles = (input_dim + BK - 1) / BK;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
        c_frag[WARP_TILE_M][WARP_TILE_N];
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        a_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::col_major>
        b_frag[WARP_TILE_N];

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i)
    {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j)
      {
        wmma::fill_fragment(c_frag[i][j], 0.0f);
      }
    }

    // Load one K tile. Full aligned 8-half vectors use cp.async; edge tiles
    // use scalar zero-padding so arbitrary prompt/output sizes remain valid.
    for (int tile_k = 0; tile_k < num_k_tiles; ++tile_k)
    {
      const int buffer = tile_k & 1;
      const int global_k = tile_k * BK + load_k;
      const bool full_k_vector = global_k + 7 < input_dim && (input_dim & 7) == 0;

      if (global_a_row < batch_size && full_k_vector)
      {
        const uint32_t dst = __cvta_generic_to_shared(
            &shared.operands.a[buffer][load_row][load_k]);
        CP_ASYNC_CG(dst, input + global_a_row * input_dim + global_k, 16);
      }
      else
      {
#pragma unroll
        for (int v = 0; v < 8; ++v)
        {
          const int k = global_k + v;
          shared.operands.a[buffer][load_row][load_k + v] =
              (global_a_row < batch_size && k < input_dim)
                  ? input[global_a_row * input_dim + k]
                  : __float2half(0.0f);
        }
      }

      if (global_w_row < output_dim && full_k_vector)
      {
        const uint32_t dst = __cvta_generic_to_shared(
            &shared.operands.weight[buffer][load_row][load_k]);
        CP_ASYNC_CG(dst, weight + global_w_row * input_dim + global_k, 16);
      }
      else
      {
#pragma unroll
        for (int v = 0; v < 8; ++v)
        {
          const int k = global_k + v;
          shared.operands.weight[buffer][load_row][load_k + v] =
              (global_w_row < output_dim && k < input_dim)
                  ? weight[global_w_row * input_dim + k]
                  : __float2half(0.0f);
        }
      }

      CP_ASYNC_COMMIT_GROUP();
      CP_ASYNC_WAIT_GROUP(0);
      __syncthreads();

#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i)
      {
        const int row = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
        wmma::load_matrix_sync(
            a_frag[i], &shared.operands.a[buffer][row][0], BK + OFFSET);
      }
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j)
      {
        const int col = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
        wmma::load_matrix_sync(
            b_frag[j], &shared.operands.weight[buffer][col][0], BK + OFFSET);
      }
#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i)
      {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j)
        {
          wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
        }
      }
      __syncthreads();
    }

    // Store fragments to shared memory first. This prevents the 16x16 WMMA
    // store from writing past the output for edge blocks and applies scale.
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i)
    {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j)
      {
        const int row = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
        const int col = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
        wmma::store_matrix_sync(&shared.c[row][col], c_frag[i][j], BN,
                                wmma::mem_row_major);
      }
    }
    __syncthreads();

    for (int index = tid; index < BM * BN; index += blockDim.x)
    {
      const int row = index / BN;
      const int col = index % BN;
      const int global_row = blockIdx.y * BM + row;
      const int global_col = blockIdx.x * BN + col;
      if (global_row < batch_size && global_col < output_dim)
      {
        output[global_row * output_dim + global_col] =
            __float2half(__half2float(shared.c[row][col]) * scale);
      }
    }
  }

#endif
  void batched_matmul_kernel_cu_pure_fp16(const tensor::Tensor &input, const tensor::Tensor &weight,
                                          const tensor::Tensor &output, int32_t batch_size,
                                          float scale, const CudaConfig *config)
  {
    CHECK(input.is_empty() == false);
    CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(input.data_type() == base::DataType::kDataTypeFp16);
    CHECK(weight.is_empty() == false && weight.dims_size() == 2);
    CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(weight.data_type() == base::DataType::kDataTypeFp16);
    CHECK(output.data_type() == base::DataType::kDataTypeFp16);
    const int32_t K = weight.get_dim(0); // output dim
    const int32_t M = weight.get_dim(1); // input dim
    const int32_t output_dim = weight.get_dim(0);
    const int32_t input_dim = weight.get_dim(1);
    CHECK(input.dims_size() == 2);
    CHECK(output.dims_size() == 2);
    CHECK_EQ(input.get_dim(0), batch_size);
    CHECK_EQ(input.get_dim(1), M);
    CHECK_EQ(output.get_dim(0), batch_size);
    CHECK_EQ(output.get_dim(1), K);

    cudaStream_t stream = config ? config->stream : nullptr;
#if BATCHED_MATMUL_FP16_BACKEND == 0
    // custom kernel
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    dim3 grid((K + WMMA_N - 1) / WMMA_N, (batch_size + WMMA_M - 1) / WMMA_M);
    batched_gemm_pure_fp16_kernel<<<grid, 32, 0, stream>>>(
        reinterpret_cast<const __half *>(input.ptr<uint16_t>()),
        reinterpret_cast<const __half *>(weight.ptr<uint16_t>()),
        reinterpret_cast<__half *>(const_cast<uint16_t *>(output.ptr<uint16_t>())),
        batch_size, M, K, scale);
#else
    dim3 block(256);
    dim3 grid((output_dim + 127) / 128, (batch_size + 127) / 128);

    batched_gemm_wmma_async_kernel<<<
        grid,
        block,
        0,
        stream>>>(
        reinterpret_cast<const half *>(
            input.ptr<uint16_t>()),

        reinterpret_cast<const half *>(
            weight.ptr<uint16_t>()),

        reinterpret_cast<half *>(
            const_cast<uint16_t *>(
                output.ptr<uint16_t>())),

        batch_size,
        output_dim,
        input_dim,
        scale);
#endif
  }
} // namespace kernel
