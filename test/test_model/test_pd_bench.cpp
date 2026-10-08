#include <glog/logging.h>
#include <gtest/gtest.h>
#include <model/qwen3.h>
#include <tensor/tensor.h>
#include "base/buffer.h"
#include <base/tick.h>
#include <cuda_runtime_api.h>
#include <array>
#include <cmath>
#include <iomanip>
#include <vector>

namespace
{
struct PDBenchResult
{
  int32_t prompt_tokens = 0;
  int32_t decode_tokens = 0;
  double prefill_seconds = 0.0;
  double decode_seconds = 0.0;
};

PDBenchResult generate(const model::Qwen3Model &model,
                       const std::vector<int32_t> &prompt_tokens,
                       int32_t max_decode_tokens)
{
  const int32_t prompt_len = static_cast<int32_t>(prompt_tokens.size());
  LOG_IF(FATAL, prompt_tokens.empty()) << "The token is empty.";

  PDBenchResult result;
  result.prompt_tokens = prompt_len;

  int32_t next = prompt_tokens.back();
  const auto &prompt_embedding = model.embedding(prompt_tokens);
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  // TTFT：完整 prompt 提交后，到得到第一个输出 token 的时间。
  pos_tensor.index<int32_t>(0) = 0;
  tensor::Tensor prompt_input = model.fill_input(pos_tensor, prompt_embedding, true);
  CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  base::nvTT.time_push("prefill");
  const auto prefill_start = std::chrono::steady_clock::now();
  auto prefill_status = model.predict(prompt_input, pos_tensor, true, next);
  CHECK(prefill_status) << "Prefill failed: " << prefill_status.get_err_msg();
  CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const auto prefill_end = std::chrono::steady_clock::now();
  base::nvTT.time_pop("");
  result.prefill_seconds =
      std::chrono::duration<double>(prefill_end - prefill_start).count();

  int32_t generated_tokens = max_decode_tokens > 0 ? 1 : 0;

  // 首 token 已包含在 TTFT 中；TPOT 只统计后续逐 token decode。
  // Benchmark 固定执行指定 token 数，不因 EOS 提前结束，保证各轮样本量一致。
  int32_t pos = prompt_len;
  std::vector<int32_t> single_token(1);
  while (generated_tokens < max_decode_tokens)
  {
    pos_tensor.index<int32_t>(0) = pos;
    single_token[0] = next;
    const auto &token_embedding = model.embedding(single_token);
    tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, false);

    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto decode_start = std::chrono::steady_clock::now();
    base::nvTT.time_push("decode");

    const auto status = model.predict(input, pos_tensor, false, next);
    CHECK(status) << "Model prediction failed: " << status.get_err_msg();
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto decode_end = std::chrono::steady_clock::now();
    base::nvTT.time_pop("");

    result.decode_seconds +=
        std::chrono::duration<double>(decode_end - decode_start).count();
    ++result.decode_tokens;
    ++generated_tokens;
    ++pos;
  }
  return result;
}

std::vector<int32_t> make_input_tokens(const std::vector<int32_t> &token_pool,
                                       int32_t target_length)
{
  CHECK(!token_pool.empty());
  CHECK_GT(target_length, 0);

  std::vector<int32_t> input_tokens;
  input_tokens.reserve(target_length);
  for (int32_t i = 0; i < target_length; ++i)
  {
    input_tokens.push_back(token_pool[static_cast<size_t>(i) % token_pool.size()]);
  }
  return input_tokens;
}
} // namespace

TEST(test_bench_pd, cuda)
{
  using namespace base;
  nvTT.time_push("all");
  mP.mem_push("af_init");
  nvTT.time_push("model.init");
  const char *checkpoint_path = "Qwen3-4B-fp16.bin"; // e.g. out/model.bin
  const char *tokenizer_path = "/media/l8w/Linux118/MODEL/LLM/Qwen/Qwen3-4B/tokenizer_config.json";
  model::Qwen3Model model(base::TokenizerType::kEncodeBpe, tokenizer_path, checkpoint_path, false);
  auto status = model.init(base::DeviceType::kDeviceCUDA);
  ASSERT_TRUE(status) << "Qwen3 model initialization failed: " << status.get_err_msg();
  const int32_t max_decode_tokens = 16;

  {
    nvTT.time_pop("");
    mP.mem_push("af_pre");

    const std::array input_lengths = {int32_t{256}, int32_t{512}, int32_t{1024},
                                      int32_t{2048}, int32_t{4096}, int32_t{8192}};
    const std::string base_text =
        "人工智能是计算机科学的一个分支，它研究如何让机器感知、推理、学习和决策。"
        "常见应用包括自然语言处理、计算机视觉、语音识别、机器人和推荐系统。"
        "深度学习使用多层神经网络从大量数据中提取特征，并完成预测或生成任务。";
    std::string long_text;
    long_text.reserve(base_text.size() * 50);
    for (int i = 0; i < 50; ++i)
    {
      long_text += base_text;
    }

    const auto all_tokens = model.encode(long_text);
    ASSERT_FALSE(all_tokens.empty());
    LOG(INFO) << "Token pool size: " << all_tokens.size();

    const auto &key_cache = model.get_buffer(model::ModelBufferType::kKeyCache);
    ASSERT_GE(key_cache.dims_size(), 2);
    const int32_t max_seq_len = key_cache.get_dim(1);

    const int warmup = 1;
    const int repeat = 2;
    const int32_t warmup_input_length =
        std::min<int32_t>(64, max_seq_len - max_decode_tokens);
    ASSERT_GT(warmup_input_length, 0);
    const auto warmup_tokens = make_input_tokens(all_tokens, warmup_input_length);

    // 预热，避免把 CUDA context、kernel 首次启动等开销算进去
    for (int i = 0; i < warmup; ++i)
    {
      const auto result = generate(model, warmup_tokens, max_decode_tokens);
      ASSERT_EQ(result.prompt_tokens, warmup_input_length);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

    nvTT.timeTable["prefill"] = microSec::zero();
    nvTT.timeTable["decode"] = microSec::zero();

    constexpr int length_column_width = 8;
    constexpr int metric_column_width = 10;
    std::cout << std::right
              << std::setw(length_column_width) << "p_length" << ','
              << std::setw(length_column_width) << "d_length" << ','
              << std::setw(metric_column_width) << "ttft_ms" << ','
              << std::setw(metric_column_width) << "p_tps" << ','
              << std::setw(metric_column_width) << "tpot_ms" << ','
              << std::setw(metric_column_width) << "d_tps" << '\n';

    for (size_t length_index = 0; length_index < input_lengths.size(); ++length_index)
    {
      const int32_t target_input = input_lengths[length_index];
      if (target_input + max_decode_tokens > max_seq_len)
      {
        LOG(WARNING) << "Skip input_tokens=" << target_input
                     << ": input + decode exceeds max_seq_len=" << max_seq_len;
        continue;
      }

      const auto input_tokens = make_input_tokens(all_tokens, target_input);
      ASSERT_EQ(static_cast<int32_t>(input_tokens.size()), target_input);

      double total_prefill_seconds = 0.0;
      double total_decode_seconds = 0.0;
      int64_t total_prefill_tokens = 0;
      int64_t total_decode_tokens = 0;

      const std::string infer_tag = "infer_" + std::to_string(target_input);
      nvTT.time_push(infer_tag);
      for (int i = 0; i < repeat; ++i)
      {
        fflush(stdout);
        const auto result = generate(model, input_tokens, max_decode_tokens);
        ASSERT_EQ(result.prompt_tokens, target_input);
        total_prefill_seconds += result.prefill_seconds;
        total_decode_seconds += result.decode_seconds;
        total_prefill_tokens += result.prompt_tokens;
        total_decode_tokens += result.decode_tokens;
        fflush(stdout);
      }
      nvTT.time_pop(infer_tag);

      const double avg_ttft_ms = total_prefill_seconds * 1000.0 / repeat;
      const double prefill_tokens_per_second = total_prefill_seconds > 0.0
                                                   ? total_prefill_tokens / total_prefill_seconds
                                                   : 0.0;
      const double avg_tpot_ms = total_decode_tokens > 0
                                     ? total_decode_seconds * 1000.0 / total_decode_tokens
                                     : 0.0;
      const double decode_tokens_per_second = total_decode_seconds > 0.0
                                                  ? total_decode_tokens / total_decode_seconds
                                                  : 0.0;

      std::cout << std::fixed << std::setprecision(3)
                << std::setw(length_column_width) << target_input << ','
                << std::setw(length_column_width) << max_decode_tokens << ','
                << std::setw(metric_column_width) << avg_ttft_ms << ','
                << std::setw(metric_column_width) << prefill_tokens_per_second << ','
                << std::setw(metric_column_width) << avg_tpot_ms << ','
                << std::setw(metric_column_width) << decode_tokens_per_second << '\n';
    }

    mP.mem_push("af_infer");
    nvTT.time_push("post");
    const __half *logits =
        model.get_buffer(model::ModelBufferType::kForwardOutput).ptr<__half>();
    ASSERT_NE(logits, nullptr);
    const auto &output = model.get_buffer(model::ModelBufferType::kForwardOutput);
    ASSERT_GT(output.size(), 0u);
    std::array<float, 8> sample{};
    const size_t sample_count = std::min(sample.size(), output.size());
    std::array<__half, 8> sample_fp16{};
    ASSERT_EQ(cudaMemcpy(sample_fp16.data(), logits, sample_count * sizeof(__half),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t i = 0; i < sample_count; ++i)
    {
      sample[i] = __half2float(sample_fp16[i]);
      ASSERT_TRUE(std::isfinite(sample[i])) << "non-finite logit at index " << i;
    }
    std::cout << "first logits:";
    for (size_t i = 0; i < sample_count; ++i)
      std::cout << ' ' << sample[i];
    std::cout << '\n';
    nvTT.time_pop("post");
    mP.mem_push("af_post");
  }
  nvTT.time_pop("all");
  // trace
  nvTT.print_avg_stats();
  // mP.print_mem_info();
}
