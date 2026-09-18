#include <glog/logging.h>
#include <gtest/gtest.h>
#include <model/qwen3.h>
#include <tensor/tensor.h>
#include "base/buffer.h"
#include <base/tick.h>
#include <cuda_runtime_api.h>
#include <array>
#include <cmath>
int32_t generate(const model::Qwen3Model &model, const std::string &sentence, int total_steps,
                 bool need_output = false)
{
  auto tokens = model.encode(sentence);
  int32_t prompt_len = tokens.size();
  LOG_IF(FATAL, tokens.empty()) << "The token is empty.";

  int32_t pos = 0;
  int32_t next = tokens.back();
  bool is_prompt = true;
  const auto &prompt_embedding = model.embedding(tokens);
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  // Prefill is submitted once as [prompt_len, hidden_size].  Its last-row
  // logits produce the first token for the decode phase.
  pos_tensor.index<int32_t>(0) = 0;
  tensor::Tensor prompt_input = model.fill_input(pos_tensor, prompt_embedding, true);
  auto prefill_status = model.predict(prompt_input, pos_tensor, true, next);
  CHECK(prefill_status) << "Prefill failed: " << prefill_status.get_err_msg();
  pos = prompt_len;
  is_prompt = false;

  std::vector<int32_t> words;
  while (pos < total_steps)
  {
    // `next` is the first token sampled from the prefill logits.  Emit it
    // before feeding it back into the model for the following position.
    if (model.is_sentence_ending(next)) break;
    words.push_back(next);
    ++pos;
    if (pos >= total_steps) break;

    pos_tensor.index<int32_t>(0) = pos;
    tokens = std::vector<int32_t>{next};
    const auto &token_embedding = model.embedding(tokens);
    tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, false);
    const auto status = model.predict(input, pos_tensor, false, next);
    CHECK(status) << "Model prediction failed: " << status.get_err_msg();
  }
  if (need_output)
  {
    printf("%s", model.decode(words).data());
    fflush(stdout);
  }
  return std::min(pos, total_steps);
}

std::string fill_template(const std::string &content)
{
  const std::string format =
      "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";
  std::string result = format;
  size_t pos = result.find("%s");
  if (pos != std::string::npos)
  {
    result.replace(pos, 2, content);
  }
  return result;
}

TEST(test_qwen3, cuda)
{
  using namespace base;
  nvTT.time_push("all");
  mP.mem_push("af_init");
  nvTT.time_push("model.build");
  const char *checkpoint_path = "Qwen3-4B-fp16.bin"; // e.g. out/model.bin
  const char *tokenizer_path = "/media/l8w/Linux118/MODEL/LLM/Qwen/Qwen3-4B/tokenizer_config.json";
  model::Qwen3Model model(base::TokenizerType::kEncodeBpe, tokenizer_path, checkpoint_path, false);
  nvTT.time_pop("");
  nvTT.time_push("model.init");
  auto status = model.init(base::DeviceType::kDeviceCUDA);
  ASSERT_TRUE(status) << "Qwen3 model initialization failed: " << status.get_err_msg();
  // Total sequence length = prompt length + generated tokens.
  const int total_steps = 32;

  {
    std::string hi = "What is AI?";
    std::cout << hi << "\n";
    const std::string &sentence = fill_template(hi);
    nvTT.time_pop("");
    mP.mem_push("af_pre");

    int warmup = 3;
    int repeat = 10;
    // 预热，避免把 CUDA context、kernel 首次启动等开销算进去
    for (int i = 0; i < warmup; ++i)
    {
      int steps = generate(model, sentence, total_steps, false);
      ASSERT_GT(steps, 0);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
    for (int i = 0; i < repeat; ++i)
    {
      nvTT.time_push("infer");

      auto start = std::chrono::steady_clock::now();
      fflush(stdout);
      int steps = generate(model, sentence, total_steps, (i == repeat - 1));
      auto end = std::chrono::steady_clock::now();
      auto duration = std::chrono::duration<double>(end - start).count();

      fflush(stdout);
      nvTT.time_pop("infer");
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
  nvTT.print_avg_stats();
  mP.print_mem_info();
}
