#include "mha_kernel.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>
#include <mma.h>

#ifndef INFER_BATCHED_MHA_BACKEND
#define INFER_BATCHED_MHA_BACKEND 1
#endif

namespace kernel
{
namespace wmma = nvcuda::wmma;
 __global__ void mha_fp16(int pos, int seq, int kvdim, int kvmul, int heads, int hs, int off, const __half *q, __half *score, __half *out, const __half *kc, const __half *vc)
 {
  int h = blockIdx.x;
  if (h >= heads)
   return;
  int ho = (h / kvmul) * hs;
  extern __shared__ float sq[];
  for (int i = threadIdx.x; i < hs; i += blockDim.x)
   sq[i] = __half2float(q[h * hs + i]);
  __syncthreads();
  for (int t = threadIdx.x; t <= pos; t += blockDim.x)
  {
   float s = 0;
   for (int i = 0; i < hs; i++)
    s += sq[i] * __half2float(kc[off + t * kvdim + ho + i]);
   score[h * seq + t] = __float2half(s / sqrtf((float)hs));
  }
  __syncthreads();
  if (threadIdx.x == 0)
  {
   float mx = -1e30f, sum = 0;
   for (int t = 0; t <= pos; t++)
    mx = fmaxf(mx, __half2float(score[h * seq + t]));
   for (int t = 0; t <= pos; t++)
   {
    float e = expf(__half2float(score[h * seq + t]) - mx);
    score[h * seq + t] = __float2half(e);
    sum += e;
   }
   for (int t = 0; t <= pos; t++)
    score[h * seq + t] = __float2half(__half2float(score[h * seq + t]) / sum);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < hs; i += blockDim.x)
  {
   float v = 0;
   for (int t = 0; t <= pos; t++)
    v += __half2float(score[h * seq + t]) * __half2float(vc[off + t * kvdim + ho + i]);
   out[h * hs + i] = __float2half(v);
  }
 }
 void mha_kernel_cu(int32_t pos, int32_t heads, int32_t layer, int32_t seq, int32_t kvdim, int32_t kvmul, int32_t hs, const tensor::Tensor &out, const tensor::Tensor &q, const tensor::Tensor &score, const tensor::Tensor &kc, const tensor::Tensor &vc, base::DeviceType, CudaConfig *cfg)
 {
  CHECK(cfg);
  CHECK(out.data_type() == base::DataType::kDataTypeFp16);
  int off = layer * seq * kvdim;
  mha_fp16<<<heads, 256, hs * sizeof(float), cfg->stream>>>(pos, seq, kvdim, kvmul, heads, hs, off, q.ptr<__half>(), const_cast<__half *>(score.ptr<__half>()), const_cast<__half *>(out.ptr<__half>()), kc.ptr<__half>(), vc.ptr<__half>());
 }
#if INFER_BATCHED_MHA_BACKEND == 0
 namespace {
 constexpr int kBatchedBlockThreads = 128;
 constexpr int kBatchedWarpSize = 32;

 __device__ __forceinline__ float batched_warp_reduce_max(float value) {
  for (int offset = kBatchedWarpSize / 2; offset > 0; offset /= 2)
   value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  return value;
 }

 __device__ __forceinline__ float batched_warp_reduce_sum(float value) {
  for (int offset = kBatchedWarpSize / 2; offset > 0; offset /= 2)
   value += __shfl_down_sync(0xffffffff, value, offset);
  return value;
 }

 __global__ void batched_mha_fp16(int start_pos, int seq_len, int max_seq, int dim, int kv_dim,
                                   int kv_mul, int heads, int hs, int layer,
                                   const __half* query, __half* score, const __half* kc,
                                   const __half* vc, __half* output) {
  const int head = blockIdx.x;
  const int row = blockIdx.y;
  if (head >= heads || row >= seq_len) return;

  const int tid = threadIdx.x;
  const int lane = tid % kBatchedWarpSize;
  const int warp = tid / kBatchedWarpSize;
  const int warp_count = blockDim.x / kBatchedWarpSize;
  const int query_pos = start_pos + row;
  const int query_offset = row * dim + head * hs;
  const int kv_head_offset = (head / kv_mul) * hs;
  const size_t layer_base = static_cast<size_t>(layer) * max_seq * kv_dim;
  const size_t score_base =
      (static_cast<size_t>(row) * heads + head) * max_seq;

  extern __shared__ float query_shared[];
  __shared__ float warp_values[kBatchedBlockThreads / kBatchedWarpSize];
  __shared__ float block_value;

  for (int d = tid; d < hs; d += blockDim.x)
   query_shared[d] = __half2float(query[query_offset + d]);
  __syncthreads();

  const float scale = rsqrtf(static_cast<float>(hs));
  float local_max = -CUDART_INF_F;
  for (int t = tid; t <= query_pos; t += blockDim.x) {
   const __half* key =
       kc + layer_base + static_cast<size_t>(t) * kv_dim + kv_head_offset;
   float dot = 0.0f;
   for (int d = 0; d < hs; ++d)
    dot += query_shared[d] * __half2float(key[d]);
   const float value = dot * scale;
   score[score_base + t] = __float2half(value);
   local_max = fmaxf(local_max, value);
  }

  local_max = batched_warp_reduce_max(local_max);
  if (lane == 0) warp_values[warp] = local_max;
  __syncthreads();
  if (warp == 0) {
   float value = lane < warp_count ? warp_values[lane] : -CUDART_INF_F;
   value = batched_warp_reduce_max(value);
   if (lane == 0) block_value = value;
  }
  __syncthreads();

  float local_sum = 0.0f;
  for (int t = tid; t <= query_pos; t += blockDim.x) {
   const float value = expf(__half2float(score[score_base + t]) - block_value);
   score[score_base + t] = __float2half(value);
   local_sum += value;
  }
  local_sum = batched_warp_reduce_sum(local_sum);
  if (lane == 0) warp_values[warp] = local_sum;
  __syncthreads();
  if (warp == 0) {
   float value = lane < warp_count ? warp_values[lane] : 0.0f;
   value = batched_warp_reduce_sum(value);
   if (lane == 0) block_value = value;
  }
  __syncthreads();

  const float inv_sum = 1.0f / block_value;
  for (int d = tid; d < hs; d += blockDim.x) {
   float value = 0.0f;
   for (int t = 0; t <= query_pos; ++t) {
    const __half* value_row =
        vc + layer_base + static_cast<size_t>(t) * kv_dim + kv_head_offset;
    value += __half2float(score[score_base + t]) * inv_sum *
             __half2float(value_row[d]);
   }
   output[static_cast<size_t>(row) * dim + head * hs + d] =
       __float2half(value);
  }
 }
 }  // namespace
#else
namespace {

constexpr int kWarpSize = 32;

__device__ __forceinline__ float warp_reduce_max(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

// 通用回退路径：一块处理一个 (query row, query head)，支持任意 head_size。
__global__ void batched_mha_prefill_generic_kernel(
    int start_pos, int seq_len, int max_seq_len, int dim, int kv_dim,
    int kv_mul, int heads, int head_size, int layer, const __half* query,
    __half* score, const __half* key_cache, const __half* value_cache,
    __half* output) {
  const int head = blockIdx.x;
  const int row = blockIdx.y;
  if (head >= heads || row >= seq_len) return;

  const int tid = threadIdx.x;
  const int lane = tid % kWarpSize;
  const int warp = tid / kWarpSize;
  const int warp_count = blockDim.x / kWarpSize;
  const int query_pos = start_pos + row;
  const int kv_head_offset = (head / kv_mul) * head_size;
  const size_t layer_base = static_cast<size_t>(layer) * max_seq_len * kv_dim;
  const size_t score_base =
      (static_cast<size_t>(row) * heads + head) * max_seq_len;

  extern __shared__ float query_shared[];
  __shared__ float warp_values[4];
  __shared__ float block_value;

  for (int d = tid; d < head_size; d += blockDim.x) {
    query_shared[d] =
        __half2float(query[static_cast<size_t>(row) * dim + head * head_size + d]);
  }
  __syncthreads();

  const float scale = rsqrtf(static_cast<float>(head_size));
  float local_max = -CUDART_INF_F;
  for (int kv_pos = tid; kv_pos <= query_pos; kv_pos += blockDim.x) {
    const __half* key = key_cache + layer_base +
                        static_cast<size_t>(kv_pos) * kv_dim + kv_head_offset;
    float dot = 0.0f;
    for (int d = 0; d < head_size; ++d) {
      dot += query_shared[d] * __half2float(key[d]);
    }
    const float value = dot * scale;
    score[score_base + kv_pos] = __float2half(value);
    local_max = fmaxf(local_max, value);
  }

  local_max = warp_reduce_max(local_max);
  if (lane == 0) warp_values[warp] = local_max;
  __syncthreads();
  if (warp == 0) {
    float value = lane < warp_count ? warp_values[lane] : -CUDART_INF_F;
    value = warp_reduce_max(value);
    if (lane == 0) block_value = value;
  }
  __syncthreads();

  float local_sum = 0.0f;
  for (int kv_pos = tid; kv_pos <= query_pos; kv_pos += blockDim.x) {
    const float value = expf(__half2float(score[score_base + kv_pos]) - block_value);
    score[score_base + kv_pos] = __float2half(value);
    local_sum += value;
  }
  local_sum = warp_reduce_sum(local_sum);
  if (lane == 0) warp_values[warp] = local_sum;
  __syncthreads();
  if (warp == 0) {
    float value = lane < warp_count ? warp_values[lane] : 0.0f;
    value = warp_reduce_sum(value);
    if (lane == 0) block_value = value;
  }
  __syncthreads();

  const float inv_sum = 1.0f / block_value;
  for (int d = tid; d < head_size; d += blockDim.x) {
    float value = 0.0f;
    for (int kv_pos = 0; kv_pos <= query_pos; ++kv_pos) {
      const __half* value_row = value_cache + layer_base +
                                static_cast<size_t>(kv_pos) * kv_dim + kv_head_offset;
      value += __half2float(score[score_base + kv_pos]) * inv_sum *
               __half2float(value_row[d]);
    }
    output[static_cast<size_t>(row) * dim + head * head_size + d] =
        __float2half(value);
  }
}

template <int HeadDim, int QueryRows, int KvCols, int NumWarps>
__global__ void batched_mha_prefill_wmma_kernel(
    int start_pos, int seq_len, int max_seq_len, int dim, int kv_dim,
    int kv_mul, int heads, int layer, const __half* query,
    const __half* key_cache, const __half* value_cache, __half* output) {
  static_assert(QueryRows == 16, "WMMA query tile must contain 16 rows");
  static_assert(KvCols == NumWarps * 16,
                "each warp must own 16 score columns");
  static_assert(HeadDim % (NumWarps * 16) == 0,
                "head dimension must divide evenly across warps");

  constexpr int kThreads = NumWarps * kWarpSize;
  constexpr int kKvColsPerWarp = KvCols / NumWarps;
  constexpr int kOutColsPerWarp = HeadDim / NumWarps;
  constexpr int kOutFragments = kOutColsPerWarp / 16;
  constexpr int kPadding = 8;
  constexpr int kKvStride = HeadDim + kPadding;

  constexpr int kQBytes = QueryRows * kKvStride * sizeof(__half);
  constexpr int kKvBytes = KvCols * kKvStride * sizeof(__half);
  constexpr int kK0Offset = kQBytes;
  constexpr int kK1Offset = kK0Offset + kKvBytes;
  constexpr int kV0Offset = kK1Offset + kKvBytes;
  constexpr int kV1Offset = kV0Offset + kKvBytes;
  constexpr int kScoreOffset = kV1Offset + kKvBytes;
  constexpr int kProbOffset =
      kScoreOffset + QueryRows * KvCols * sizeof(float);
  constexpr int kOutputOffset =
      kProbOffset + QueryRows * KvCols * sizeof(__half);
  constexpr int kWarpOutputOffset =
      kOutputOffset + QueryRows * HeadDim * sizeof(float);
  constexpr int kRowMaxOffset =
      kWarpOutputOffset +
      NumWarps * QueryRows * kOutColsPerWarp * sizeof(float);
  constexpr int kRowSumOffset = kRowMaxOffset + QueryRows * sizeof(float);
  constexpr int kAlphaOffset = kRowSumOffset + QueryRows * sizeof(float);

  extern __shared__ char shared_bytes[];
  __half* query_shared = reinterpret_cast<__half*>(shared_bytes);
  __half* key_buffers[2] = {
      reinterpret_cast<__half*>(shared_bytes + kK0Offset),
      reinterpret_cast<__half*>(shared_bytes + kK1Offset)};
  __half* value_buffers[2] = {
      reinterpret_cast<__half*>(shared_bytes + kV0Offset),
      reinterpret_cast<__half*>(shared_bytes + kV1Offset)};
  float* score_shared = reinterpret_cast<float*>(shared_bytes + kScoreOffset);
  __half* prob_shared = reinterpret_cast<__half*>(shared_bytes + kProbOffset);
  float* output_shared = reinterpret_cast<float*>(shared_bytes + kOutputOffset);
  float* warp_output = reinterpret_cast<float*>(shared_bytes + kWarpOutputOffset);
  float* row_max = reinterpret_cast<float*>(shared_bytes + kRowMaxOffset);
  float* row_sum = reinterpret_cast<float*>(shared_bytes + kRowSumOffset);
  float* alpha = reinterpret_cast<float*>(shared_bytes + kAlphaOffset);

  const int tid = threadIdx.x;
  const int warp = tid / kWarpSize;
  const int head = blockIdx.y;
  const int query_tile_start = blockIdx.x * QueryRows;
  if (head >= heads || query_tile_start >= seq_len) return;

  const int kv_head_offset = (head / kv_mul) * HeadDim;
  const size_t layer_base = static_cast<size_t>(layer) * max_seq_len * kv_dim;
  const float scale = rsqrtf(static_cast<float>(HeadDim));

  constexpr int kKvBufferElements = KvCols * kKvStride;
  for (int i = tid; i < kKvBufferElements; i += kThreads) {
    key_buffers[0][i] = __float2half(0.0f);
    key_buffers[1][i] = __float2half(0.0f);
    value_buffers[0][i] = __float2half(0.0f);
    value_buffers[1][i] = __float2half(0.0f);
  }
  for (int i = tid; i < QueryRows * HeadDim; i += kThreads) {
    output_shared[i] = 0.0f;
  }
  if (tid < QueryRows) {
    row_max[tid] = -CUDART_INF_F;
    row_sum[tid] = 0.0f;
  }

  for (int i = tid; i < QueryRows * HeadDim; i += kThreads) {
    const int row = i / HeadDim;
    const int d = i % HeadDim;
    const int query_row = query_tile_start + row;
    query_shared[row * kKvStride + d] =
        query_row < seq_len
            ? query[static_cast<size_t>(query_row) * dim + head * HeadDim + d]
            : __float2half(0.0f);
  }
  __syncthreads();

  const int query_end =
      min(start_pos + seq_len, start_pos + query_tile_start + QueryRows);
  const int kv_tile_count = (query_end + KvCols - 1) / KvCols;
  constexpr int kAsyncCopies = KvCols * HeadDim / 8;

  for (int i = tid; i < kAsyncCopies; i += kThreads) {
    const int row = i / (HeadDim / 8);
    const int d = (i % (HeadDim / 8)) * 8;
    if (row < max_seq_len) {
      const size_t global_offset = layer_base +
          static_cast<size_t>(row) * kv_dim + kv_head_offset + d;
      const unsigned int key_shared_addr =
          __cvta_generic_to_shared(key_buffers[0] + row * kKvStride + d);
      const unsigned int value_shared_addr =
          __cvta_generic_to_shared(value_buffers[0] + row * kKvStride + d);
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                   :: "r"(key_shared_addr),
                      "l"(reinterpret_cast<unsigned long long>(key_cache + global_offset))
                   : "memory");
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                   :: "r"(value_shared_addr),
                      "l"(reinterpret_cast<unsigned long long>(value_cache + global_offset))
                   : "memory");
    }
  }
  asm volatile("cp.async.commit_group;" ::: "memory");

  for (int kv_tile = 0; kv_tile < kv_tile_count; ++kv_tile) {
    const int kv_start = kv_tile * KvCols;
    const int current_buffer = kv_tile & 1;
    __half* key_current = key_buffers[current_buffer];
    __half* value_current = value_buffers[current_buffer];

    if (kv_tile + 1 < kv_tile_count) {
      const int next_start = (kv_tile + 1) * KvCols;
      const int next_buffer = 1 - current_buffer;
      for (int i = tid; i < kAsyncCopies; i += kThreads) {
        const int row = i / (HeadDim / 8);
        const int d = (i % (HeadDim / 8)) * 8;
        const int kv_pos = next_start + row;
        if (kv_pos < max_seq_len) {
          const size_t global_offset = layer_base +
              static_cast<size_t>(kv_pos) * kv_dim + kv_head_offset + d;
          const unsigned int key_shared_addr = __cvta_generic_to_shared(
              key_buffers[next_buffer] + row * kKvStride + d);
          const unsigned int value_shared_addr = __cvta_generic_to_shared(
              value_buffers[next_buffer] + row * kKvStride + d);
          asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                       :: "r"(key_shared_addr),
                          "l"(reinterpret_cast<unsigned long long>(key_cache + global_offset))
                       : "memory");
          asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                       :: "r"(value_shared_addr),
                          "l"(reinterpret_cast<unsigned long long>(value_cache + global_offset))
                       : "memory");
        }
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
      asm volatile("cp.async.wait_group 1;" ::: "memory");
    } else {
      asm volatile("cp.async.wait_group 0;" ::: "memory");
    }
    __syncthreads();

    {
      const int score_col = warp * kKvColsPerWarp;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>
          query_fragment;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>
          key_fragment;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float>
          score_fragment;
      wmma::fill_fragment(score_fragment, 0.0f);
#pragma unroll
      for (int d = 0; d < HeadDim; d += 16) {
        wmma::load_matrix_sync(query_fragment, query_shared + d, kKvStride);
        wmma::load_matrix_sync(
            key_fragment, key_current + score_col * kKvStride + d, kKvStride);
        wmma::mma_sync(score_fragment, query_fragment, key_fragment, score_fragment);
      }
      wmma::store_matrix_sync(score_shared + score_col, score_fragment, KvCols,
                              wmma::mem_row_major);
    }
    __syncthreads();

    for (int i = tid; i < QueryRows * KvCols; i += kThreads) {
      const int row = i / KvCols;
      const int col = i % KvCols;
      const int query_row = query_tile_start + row;
      const int query_pos = start_pos + query_row;
      const int kv_pos = kv_start + col;
      float value = score_shared[i] * scale;
      if (query_row >= seq_len || kv_pos > query_pos || kv_pos >= max_seq_len) {
        value = -CUDART_INF_F;
      }
      score_shared[i] = value;
    }
    __syncthreads();

    {
      constexpr int kThreadsPerRow = kThreads / QueryRows;
      constexpr int kColsPerThread = KvCols / kThreadsPerRow;
      const int row = tid / kThreadsPerRow;
      const int col_start = (tid % kThreadsPerRow) * kColsPerThread;
      const bool valid_row = query_tile_start + row < seq_len;
      const float old_max = row_max[row];
      float new_max = old_max;
      if (valid_row) {
#pragma unroll
        for (int col = col_start; col < col_start + kColsPerThread; ++col) {
          new_max = fmaxf(new_max, score_shared[row * KvCols + col]);
        }
      }
      new_max = fmaxf(new_max, __shfl_xor_sync(0xffffffff, new_max, 4));
      new_max = fmaxf(new_max, __shfl_xor_sync(0xffffffff, new_max, 2));
      new_max = fmaxf(new_max, __shfl_xor_sync(0xffffffff, new_max, 1));
      const float row_alpha = valid_row ? expf(old_max - new_max) : 1.0f;

      float tile_sum = 0.0f;
      if (valid_row) {
#pragma unroll
        for (int col = col_start; col < col_start + kColsPerThread; ++col) {
          const float probability =
              expf(score_shared[row * KvCols + col] - new_max);
          prob_shared[row * KvCols + col] = __float2half(probability);
          tile_sum += probability;
        }
      } else {
#pragma unroll
        for (int col = col_start; col < col_start + kColsPerThread; ++col) {
          prob_shared[row * KvCols + col] = __float2half(0.0f);
        }
      }
      tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, 4);
      tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, 2);
      tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, 1);
      if (tid % kThreadsPerRow == 0) {
        row_max[row] = new_max;
        row_sum[row] = row_sum[row] * row_alpha + tile_sum;
        alpha[row] = valid_row ? row_alpha : 1.0f;
      }
    }
    __syncthreads();

    for (int i = tid; i < QueryRows * HeadDim; i += kThreads) {
      output_shared[i] *= alpha[i / HeadDim];
    }
    __syncthreads();

    {
      const int output_col = warp * kOutColsPerWarp;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>
          prob_fragment;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major>
          value_fragment[kOutFragments];
      wmma::fragment<wmma::accumulator, 16, 16, 16, float>
          output_fragment[kOutFragments];
#pragma unroll
      for (int fragment = 0; fragment < kOutFragments; ++fragment) {
        wmma::fill_fragment(output_fragment[fragment], 0.0f);
      }
#pragma unroll
      for (int k = 0; k < KvCols; k += 16) {
        wmma::load_matrix_sync(prob_fragment, prob_shared + k, KvCols);
#pragma unroll
        for (int fragment = 0; fragment < kOutFragments; ++fragment) {
          wmma::load_matrix_sync(
              value_fragment[fragment],
              value_current + k * kKvStride + output_col + fragment * 16,
              kKvStride);
          wmma::mma_sync(output_fragment[fragment], prob_fragment,
                         value_fragment[fragment], output_fragment[fragment]);
        }
      }
#pragma unroll
      for (int fragment = 0; fragment < kOutFragments; ++fragment) {
        wmma::store_matrix_sync(
            warp_output +
                (warp * QueryRows) * kOutColsPerWarp + fragment * 16,
            output_fragment[fragment], kOutColsPerWarp, wmma::mem_row_major);
      }
    }
    __syncthreads();

    for (int i = tid; i < QueryRows * HeadDim; i += kThreads) {
      const int row = i / HeadDim;
      const int d = i % HeadDim;
      const int owner_warp = d / kOutColsPerWarp;
      const int owner_col = d % kOutColsPerWarp;
      output_shared[i] +=
          warp_output[(owner_warp * QueryRows + row) * kOutColsPerWarp + owner_col];
    }
    __syncthreads();
  }

  for (int i = tid; i < QueryRows * HeadDim; i += kThreads) {
    const int row = i / HeadDim;
    const int d = i % HeadDim;
    const int query_row = query_tile_start + row;
    if (query_row < seq_len) {
      const float inv_sum = row_sum[row] > 0.0f ? 1.0f / row_sum[row] : 0.0f;
      output[static_cast<size_t>(query_row) * dim + head * HeadDim + d] =
          __float2half(output_shared[i] * inv_sum);
    }
  }
}

constexpr int kPrefillHeadDim = 128;
constexpr int kPrefillQueryRows = 16;
constexpr int kPrefillKvCols = 64;
constexpr int kPrefillWarps = 4;
constexpr int kPrefillPadding = 8;
constexpr int kPrefillKvStride = kPrefillHeadDim + kPrefillPadding;
constexpr int kPrefillSharedBytes =
    kPrefillQueryRows * kPrefillKvStride * sizeof(__half) +
    4 * kPrefillKvCols * kPrefillKvStride * sizeof(__half) +
    kPrefillQueryRows * kPrefillKvCols * sizeof(float) +
    kPrefillQueryRows * kPrefillKvCols * sizeof(__half) +
    kPrefillQueryRows * kPrefillHeadDim * sizeof(float) +
    kPrefillWarps * kPrefillQueryRows *
        (kPrefillHeadDim / kPrefillWarps) * sizeof(float) +
    kPrefillQueryRows * 3 * sizeof(float);

}  // namespace
#endif
 void batched_mha_kernel_cu(int32_t start_pos, int32_t seq_len, int32_t head_num,
                            int32_t layer_index, int32_t max_seq_len, int32_t dim,
                            int32_t kv_dim, int32_t kv_mul, int32_t head_size,
                            const tensor::Tensor& output, const tensor::Tensor& query,
                            const tensor::Tensor& score, const tensor::Tensor& key_cache,
                            const tensor::Tensor& value_cache, CudaConfig* config) {
#if INFER_BATCHED_MHA_BACKEND == 0
  dim3 grid(head_num, seq_len);
  batched_mha_fp16<<<grid, kBatchedBlockThreads,
                      head_size * sizeof(float), config->stream>>>(
      start_pos, seq_len, max_seq_len, dim, kv_dim, kv_mul, head_num, head_size,
      layer_index, query.ptr<__half>(), const_cast<__half*>(score.ptr<__half>()),
      key_cache.ptr<__half>(), value_cache.ptr<__half>(), const_cast<__half*>(output.ptr<__half>()));
#else
  CHECK(config);
  CHECK(output.data_type() == base::DataType::kDataTypeFp16);
  CHECK(query.data_type() == base::DataType::kDataTypeFp16);
  CHECK(score.data_type() == base::DataType::kDataTypeFp16);
  CHECK(key_cache.data_type() == base::DataType::kDataTypeFp16);
  CHECK(value_cache.data_type() == base::DataType::kDataTypeFp16);
  CHECK_GE(start_pos, 0);
  CHECK_GE(layer_index, 0);
  CHECK_GT(seq_len, 0);
  CHECK_GT(max_seq_len, 0);
  CHECK_LE(start_pos + seq_len, max_seq_len);
  CHECK_GT(head_num, 0);
  CHECK_GT(head_size, 0);
  CHECK_GT(kv_dim, 0);
  CHECK_GT(kv_mul, 0);
  CHECK_EQ(dim, head_num * head_size);
  CHECK_EQ(kv_dim % head_size, 0);
  CHECK_LE((head_num + kv_mul - 1) / kv_mul, kv_dim / head_size);
  CHECK_GE(output.size(), static_cast<size_t>(seq_len) * dim);
  CHECK_GE(query.size(), static_cast<size_t>(seq_len) * dim);
  CHECK_GE(score.size(), static_cast<size_t>(seq_len) * head_num * max_seq_len);
  const size_t cache_size =
      static_cast<size_t>(layer_index + 1) * max_seq_len * kv_dim;
  CHECK_GE(key_cache.size(), cache_size);
  CHECK_GE(value_cache.size(), cache_size);

  if (head_size == kPrefillHeadDim) {
    static bool shared_memory_configured = false;
    if (!shared_memory_configured) {
      const cudaError_t status = cudaFuncSetAttribute(
          batched_mha_prefill_wmma_kernel<
              kPrefillHeadDim, kPrefillQueryRows, kPrefillKvCols,
              kPrefillWarps>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, kPrefillSharedBytes);
      CHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
      shared_memory_configured = true;
    }
    dim3 grid((seq_len + kPrefillQueryRows - 1) / kPrefillQueryRows,
              head_num);
    batched_mha_prefill_wmma_kernel<
        kPrefillHeadDim, kPrefillQueryRows, kPrefillKvCols, kPrefillWarps>
        <<<grid, kPrefillWarps * kWarpSize, kPrefillSharedBytes,
           config->stream>>>(
            start_pos, seq_len, max_seq_len, dim, kv_dim, kv_mul, head_num,
            layer_index, query.ptr<__half>(), key_cache.ptr<__half>(),
            value_cache.ptr<__half>(), const_cast<__half*>(output.ptr<__half>()));
  } else {
    dim3 grid(head_num, seq_len);
    batched_mha_prefill_generic_kernel<<<grid, 128,
                                         head_size * sizeof(float),
                                         config->stream>>>(
        start_pos, seq_len, max_seq_len, dim, kv_dim, kv_mul, head_num,
        head_size, layer_index, query.ptr<__half>(),
        const_cast<__half*>(score.ptr<__half>()), key_cache.ptr<__half>(),
        value_cache.ptr<__half>(), const_cast<__half*>(output.ptr<__half>()));
  }
#endif
}
}
