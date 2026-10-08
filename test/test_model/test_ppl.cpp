#include <gtest/gtest.h>
#include <model/qwen3.h>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kBlockSize = 1024;

class TokenReader {
 public:
  explicit TokenReader(const char* mode) {
    const std::string command =
        "/home/l8w/miniconda3/envs/cp312/bin/python test/test_model/ppl_tokens.py " +
        std::string(mode);
    pipe_ = popen(command.c_str(), "r");
    if (!pipe_) throw std::runtime_error("无法启动本地数据分词脚本");
  }

  ~TokenReader() {
    if (pipe_) pclose(pipe_);
  }

  uint32_t read_size() {
    uint32_t size = 0;
    read(&size, sizeof(size));
    return size;
  }

  std::string read_text() {
    const uint32_t size = read_size();
    if (size > 4096) throw std::runtime_error("句子长度异常");
    std::string text(size, '\0');
    read(text.data(), size);
    return text;
  }

  std::vector<int32_t> read_ids() {
    const uint32_t size = read_size();
    if (size < 2 || size > 10000000) throw std::runtime_error("token 数量异常");
    std::vector<int32_t> ids(size);
    read(ids.data(), ids.size() * sizeof(int32_t));
    return ids;
  }

  void finish() {
    const int status = pclose(pipe_);
    pipe_ = nullptr;
    if (status != 0) throw std::runtime_error("本地数据分词脚本执行失败");
  }

 private:
  void read(void* data, size_t bytes) {
    if (fread(data, 1, bytes, pipe_) != bytes)
      throw std::runtime_error("分词脚本输出不完整，请检查 Python 报错");
  }

  FILE* pipe_ = nullptr;
};

struct Result {
  size_t tokens = 0;
  double nll = 0;

  void print(const std::string& name, const std::string& sentence = "") const {
    std::cout << name << ',' << tokens << ',' << std::fixed << std::setprecision(6)
              << nll / tokens << ',' << std::exp(nll / tokens) << ",\"" << sentence
              << "\"\n" << std::flush;
  }
};

// 每个块从位置 0 独立 Prefill；块间只重叠一个 token，不携带 KV cache。
Result evaluate(model::Qwen3Model& model, const std::vector<int32_t>& ids,
                tensor::Tensor& logits, std::vector<__half>& host_logits,
                size_t max_blocks = 0) {
  Result result;
  const size_t vocab = model.get_buffer(model::ModelBufferType::kForwardOutput).size();
  auto pos = model.get_buffer(model::ModelBufferType::kInputPos);
  size_t blocks = 0;
  for (size_t start = 0; start + 1 < ids.size(); start += kBlockSize - 1) {
    if (max_blocks && blocks >= max_blocks) break;
    const size_t count = std::min<size_t>(kBlockSize, ids.size() - start);
    std::vector<int32_t> block(ids.begin() + start, ids.begin() + start + count);
    pos.index<int32_t>(0) = 0;
    const auto embeddings = model.embedding(block);
    const auto input = model.fill_input(pos, embeddings, true);
    logits.reshape({static_cast<int32_t>(count), static_cast<int32_t>(vocab)});
    const auto status = model.prefill(input, static_cast<int32_t>(count), 0, &logits);
    if (!status) throw std::runtime_error("Prefill 失败: " + status.get_err_msg());
    const auto cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess)
      throw std::runtime_error(std::string("CUDA 错误: ") + cudaGetErrorString(cuda_status));
    const auto copy_status = cudaMemcpy(host_logits.data(), logits.ptr<__half>(),
                                        count * vocab * sizeof(__half), cudaMemcpyDeviceToHost);
    if (copy_status != cudaSuccess)
      throw std::runtime_error(std::string("logits 拷贝失败: ") + cudaGetErrorString(copy_status));

    for (size_t row = 0; row + 1 < count; ++row) {
      const __half* scores = host_logits.data() + row * vocab;
      float maximum = -INFINITY;
      for (size_t col = 0; col < vocab; ++col)
        maximum = std::max(maximum, __half2float(scores[col]));
      double sum = 0;
      for (size_t col = 0; col < vocab; ++col)
        sum += std::exp(static_cast<double>(__half2float(scores[col]) - maximum));
      const int32_t label = block[row + 1];
      if (label < 0 || static_cast<size_t>(label) >= vocab)
        throw std::runtime_error("目标 token 超出词表范围");
      result.nll += maximum + std::log(sum) - __half2float(scores[label]);
      ++result.tokens;
    }
    ++blocks;
    if (blocks % 10 == 0) std::cerr << "WikiText 已完成 " << blocks << " 块\n";
  }
  return result;
}

}  // namespace

TEST(test_ppl, cuda) {
  model::Qwen3Model model(
      base::TokenizerType::kEncodeBpe,
      "/media/l8w/Linux118/MODEL/LLM/Qwen/Qwen3-4B/tokenizer_config.json",
      "Qwen3-4B-fp16.bin", false);
  const auto status = model.init(base::DeviceType::kDeviceCUDA);
  ASSERT_TRUE(status) << status.get_err_msg();

  const size_t vocab = model.get_buffer(model::ModelBufferType::kForwardOutput).size();
  tensor::Tensor logits(base::DataType::kDataTypeFp16, kBlockSize,
                        static_cast<int32_t>(vocab), true,
                        base::CUDADeviceAllocatorFactory::get_instance());
  std::vector<__half> host_logits(kBlockSize * vocab);

  std::cout << "index,tokens,mean_nll,perplexity,text\n";
  Result overall;
  TokenReader sentences("sentences");
  ASSERT_EQ(sentences.read_size(), 3u);
  for (int index = 1; index <= 3; ++index) {
    const auto sentence = sentences.read_text();
    const auto ids = sentences.read_ids();
    const auto result = evaluate(model, ids, logits, host_logits);
    result.print(std::to_string(index), sentence);
    overall.tokens += result.tokens;
    overall.nll += result.nll;
  }
  sentences.finish();
  overall.print("overall");

  TokenReader wikitext("wikitext");
  const auto ids = wikitext.read_ids();
  wikitext.finish();
  size_t max_blocks = 0;
  if (const char* limit = std::getenv("PPL_MAX_BLOCKS"))
    max_blocks = std::stoul(limit);
  const auto result = evaluate(model, ids, logits, host_logits, max_blocks);
  ASSERT_GT(result.tokens, 0u);
  if (max_blocks && result.tokens != ids.size() - 1)
    result.print("wikitext-2-test-partial");
  else
    result.print("wikitext-2-test");
}
