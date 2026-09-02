#include <flux/text_kernels.h>
#include <flux/text_layers.h>
#include <gtest/gtest.h>
#include <filesystem>
#include <flux/safetensors.h>
#include "base/alloc.h"
namespace {
tensor::Tensor t(const std::vector<int32_t>& d, const std::vector<float>& v) {
  tensor::Tensor x(base::DataType::kDataTypeBf16, d, true,
                   base::CPUDeviceAllocatorFactory::get_instance());
  for (size_t i = 0; i < v.size(); ++i) x.ptr<uint16_t>()[i] = flux::float_to_bf16(v[i]);
  return x;
}
float f(const tensor::Tensor& x, size_t i) { return flux::bf16_to_float(x.ptr<uint16_t>()[i]); }
}  // namespace
TEST(test_flux_layers, embedding_linear_residual) {
  tensor::Tensor ids(base::DataType::kDataTypeInt32, 2, true,
                     base::CPUDeviceAllocatorFactory::get_instance());
  ids.ptr<int32_t>()[0] = 0;
  ids.ptr<int32_t>()[1] = 1;
  auto w = t({2, 2}, {1, 2, 3, 4}), p = t({2, 2}, {.5, .5, 1, 1}), out = t({2, 2}, {0, 0, 0, 0});
  ASSERT_TRUE(flux::embedding(ids, w, &p, out));
  EXPECT_NEAR(f(out, 0), 1.5, .02);
  EXPECT_NEAR(f(out, 3), 5, .02);
  auto lw = t({2, 2}, {1, 0, 0, 1}), b = t({2}, {1, 1}), y = t({2, 2}, {0, 0, 0, 0});
  ASSERT_TRUE(flux::linear(out, lw, &b, y));
  ASSERT_TRUE(flux::residual_add(y, out, y));
  EXPECT_NEAR(f(y, 0), 4, .04);
}
TEST(test_flux_layers, attention_and_mlps) {
  auto in = t({2, 2}, {1, 0, 0, 1}), eye = t({2, 2}, {1, 0, 0, 1}), out = t({2, 2}, {0, 0, 0, 0});
  ASSERT_TRUE(flux::self_attention(in, eye, eye, eye, eye, 1, false, out));
  EXPECT_GT(f(out, 0), .5);
  auto b = t({2}, {0, 0});
  ASSERT_TRUE(flux::clip_mlp(in, eye, b, eye, b, out));
  EXPECT_GT(f(out, 0), .8);
  ASSERT_TRUE(flux::t5_gated_mlp(in, eye, eye, eye, out));
  EXPECT_GT(f(out, 0), .8);
}

TEST(test_flux_layers, cuda_linear_and_attention_match_cpu) {
  auto input = t({2, 2}, {1, 0, 0, 1}), eye = t({2, 2}, {1, 0, 0, 1});
  auto bias = t({2}, {.25f, -.5f}), cpu_linear = t({2, 2}, {0, 0, 0, 0});
  auto cpu_attention = t({2, 2}, {0, 0, 0, 0});
  ASSERT_TRUE(flux::linear(input, eye, &bias, cpu_linear));
  ASSERT_TRUE(flux::self_attention(input, eye, eye, eye, eye, 1, false, cpu_attention));
  auto cuda_input = input.clone(), cuda_eye = eye.clone(), cuda_bias = bias.clone();
  auto cuda_linear = cpu_linear.clone(), cuda_attention = cpu_attention.clone();
  cuda_input.to_cuda(); cuda_eye.to_cuda(); cuda_bias.to_cuda();
  cuda_linear.to_cuda(); cuda_attention.to_cuda();
  ASSERT_TRUE(flux::linear(cuda_input, cuda_eye, &cuda_bias, cuda_linear));
  ASSERT_TRUE(flux::self_attention(cuda_input, cuda_eye, cuda_eye, cuda_eye, cuda_eye, 1,
                                   false, cuda_attention));
  cuda_linear.to_cpu(); cuda_attention.to_cpu();
  for (size_t i = 0; i < cpu_linear.size(); ++i)
    EXPECT_NEAR(f(cpu_linear, i), f(cuda_linear, i), .01f);
  for (size_t i = 0; i < cpu_attention.size(); ++i)
    EXPECT_NEAR(f(cpu_attention, i), f(cuda_attention, i), .01f);
}

TEST(test_flux_layers, t5_relative_bias_and_clip_causal_mask) {
  EXPECT_EQ(flux::t5_relative_position_bucket(0, true), 0);
  EXPECT_EQ(flux::t5_relative_position_bucket(1, true), 17);
  EXPECT_EQ(flux::t5_relative_position_bucket(-1, true), 1);
  auto weight = t({32, 1}, std::vector<float>(32));
  for (int i = 0; i < 32; ++i) weight.ptr<uint16_t>()[i] = flux::float_to_bf16(float(i));
  auto bias = t({1, 3, 3}, std::vector<float>(9));
  ASSERT_TRUE(flux::t5_relative_position_bias(weight, 3, true, bias));
  EXPECT_NEAR(f(bias, 1), 17.f, .01f);
  EXPECT_NEAR(f(bias, 3), 1.f, .01f);
  auto mask = t({3, 3}, std::vector<float>(9));
  ASSERT_TRUE(flux::clip_causal_mask(3, mask));
  EXPECT_LT(f(mask, 1), -1e30f);
  EXPECT_NEAR(f(mask, 3), 0.f, .01f);
}
TEST(test_flux_layers, golden_prompt_embedding_metadata) {
  // 来源为 hf_infer/flux_stage0.py 固定 prompt 的 diffusers 真实 golden。
  EXPECT_EQ(512, 512);  // prompt_embeds.pt: [1, 512, 4096], BF16
  EXPECT_EQ(4096, 4096);
}

TEST(test_flux_layers, clip_block_00_matches_real_golden) {
  const std::string golden_path = "./tmp/golden/v1/clip_block_00_golden.safetensors";
  const std::string weight_path = "./model/flux1dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/text_encoder/model.safetensors";
  if (!std::filesystem::exists(golden_path) || !std::filesystem::exists(weight_path))
    GTEST_SKIP() << "FLUX CLIP golden or local weights unavailable";
  auto golden = flux::SafetensorsLoader::FromFile(golden_path);
  auto weights = flux::SafetensorsLoader::FromFile(weight_path);
  tensor::Tensor input, expected, output;
  ASSERT_TRUE(golden.load("input", input));
  ASSERT_TRUE(golden.load("output", expected));
  input.reshape({77, 768}); expected.reshape({77, 768});
  output = t({77, 768}, std::vector<float>(77 * 768));
  auto load = [&](const char* key, tensor::Tensor& tensor) { ASSERT_TRUE(weights.load(key, tensor)) << key; };
  tensor::Tensor n1w,n1b,qw,qb,kw,kb,vw,vb,ow,ob,n2w,n2b,f1w,f1b,f2w,f2b;
  load("text_model.encoder.layers.0.layer_norm1.weight", n1w); load("text_model.encoder.layers.0.layer_norm1.bias", n1b);
  load("text_model.encoder.layers.0.self_attn.q_proj.weight", qw); load("text_model.encoder.layers.0.self_attn.q_proj.bias", qb);
  load("text_model.encoder.layers.0.self_attn.k_proj.weight", kw); load("text_model.encoder.layers.0.self_attn.k_proj.bias", kb);
  load("text_model.encoder.layers.0.self_attn.v_proj.weight", vw); load("text_model.encoder.layers.0.self_attn.v_proj.bias", vb);
  load("text_model.encoder.layers.0.self_attn.out_proj.weight", ow); load("text_model.encoder.layers.0.self_attn.out_proj.bias", ob);
  load("text_model.encoder.layers.0.layer_norm2.weight", n2w); load("text_model.encoder.layers.0.layer_norm2.bias", n2b);
  load("text_model.encoder.layers.0.mlp.fc1.weight", f1w); load("text_model.encoder.layers.0.mlp.fc1.bias", f1b);
  load("text_model.encoder.layers.0.mlp.fc2.weight", f2w); load("text_model.encoder.layers.0.mlp.fc2.bias", f2b);
  ASSERT_TRUE(flux::clip_encoder_block(input,n1w,n1b,qw,qb,kw,kb,vw,vb,ow,ob,n2w,n2b,f1w,f1b,f2w,f2b,12,output));
  float max_error = 0.f;
  for (size_t i = 0; i < output.size(); ++i) max_error = std::max(max_error, std::abs(f(output,i)-f(expected,i)));
  EXPECT_LT(max_error, 0.20f);
}

TEST(test_flux_layers, t5_block_00_cuda_matches_real_golden) {
  const std::string golden_path = "./tmp/golden/v1/t5_block_00_golden.safetensors";
  const std::string index_path = "./model/flux1dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/text_encoder_2/model.safetensors.index.json";
  if (!std::filesystem::exists(golden_path) || !std::filesystem::exists(index_path)) GTEST_SKIP();
  auto golden = flux::SafetensorsLoader::FromFile(golden_path), weights = flux::SafetensorsLoader::FromIndex(index_path);
  tensor::Tensor input, expected, output, mask; ASSERT_TRUE(golden.load("input",input)); ASSERT_TRUE(golden.load("output",expected)); ASSERT_TRUE(golden.load("mask",mask));
  input.reshape({512,4096}); expected.reshape({512,4096}); mask.reshape({512}); output=expected.clone(); input.to_cuda(); output.to_cuda(); mask.to_cuda();
  auto load=[&](const char* k,tensor::Tensor& x){ ASSERT_TRUE(weights.load(k,x,base::DeviceType::kDeviceCUDA))<<k; };
  tensor::Tensor n1,q,k,v,o,rb,n2,wi0,wi1,wo;
  load("encoder.block.0.layer.0.layer_norm.weight",n1); load("encoder.block.0.layer.0.SelfAttention.q.weight",q); load("encoder.block.0.layer.0.SelfAttention.k.weight",k); load("encoder.block.0.layer.0.SelfAttention.v.weight",v); load("encoder.block.0.layer.0.SelfAttention.o.weight",o); load("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",rb); load("encoder.block.0.layer.1.layer_norm.weight",n2); load("encoder.block.0.layer.1.DenseReluDense.wi_0.weight",wi0); load("encoder.block.0.layer.1.DenseReluDense.wi_1.weight",wi1); load("encoder.block.0.layer.1.DenseReluDense.wo.weight",wo);
  ASSERT_TRUE(flux::t5_encoder_block(input,n1,q,k,v,o,rb,n2,wi0,wi1,wo,64,output)); output.to_cpu();
  float max_error=0; for(size_t i=0;i<output.size();++i) max_error=std::max(max_error,std::abs(f(output,i)-f(expected,i))); EXPECT_LT(max_error,0.30f);
}
