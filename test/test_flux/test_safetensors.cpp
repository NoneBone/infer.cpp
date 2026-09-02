#include <fstream>
#include <filesystem>
#include <gtest/gtest.h>
#include <flux/safetensors.h>

namespace {
void write_safetensors(const std::string& path) {
  const std::string header =
      R"({"bf16":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},"fp32":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}})";
  std::ofstream file(path, std::ios::binary);
  const uint64_t size = header.size();
  const uint16_t bf16[] = {0x3f80, 0x4000, 0x4040, 0x4080};
  const float fp32[] = {1.5f, -2.0f};
  file.write(reinterpret_cast<const char*>(&size), sizeof(size));
  file.write(header.data(), header.size());
  file.write(reinterpret_cast<const char*>(bf16), sizeof(bf16));
  file.write(reinterpret_cast<const char*>(fp32), sizeof(fp32));
}
}  // namespace

TEST(test_flux_safetensors, single_file_cpu_load_and_weight_map) {
  std::filesystem::create_directories("./tmp");
  const std::string path = "./tmp/test_flux.safetensors";
  write_safetensors(path);

  auto loader = flux::SafetensorsLoader::FromFile(path);
  ASSERT_EQ(loader.size(), 2);
  ASSERT_TRUE(loader.has("bf16"));
  ASSERT_EQ(loader.count_prefix(""), 2);
  ASSERT_EQ(loader.info("bf16").shape, (std::vector<int32_t>{2, 2}));

  flux::FluxWeightMap weights;
  weights.add("bf16", base::DataType::kDataTypeBf16, {2, 2});
  ASSERT_TRUE(weights.validate(loader));
  tensor::Tensor tensor;
  ASSERT_TRUE(weights.load(loader, "bf16", tensor));
  ASSERT_EQ(tensor.data_type(), base::DataType::kDataTypeBf16);
  ASSERT_EQ(tensor.dims(), (std::vector<int32_t>{2, 2}));
  ASSERT_EQ(tensor.ptr<uint16_t>()[0], 0x3f80);
  ASSERT_EQ(tensor.ptr<uint16_t>()[3], 0x4080);
}

TEST(test_flux_safetensors, sharded_index_load) {
  std::filesystem::create_directories("./tmp/flux_index");
  const std::string shard = "./tmp/flux_index/model-00001-of-00001.safetensors";
  write_safetensors(shard);
  std::ofstream index("./tmp/flux_index/model.safetensors.index.json");
  index << R"({"weight_map":{"fp32":"model-00001-of-00001.safetensors"}})";
  index.close();

  auto loader = flux::SafetensorsLoader::FromIndex("./tmp/flux_index/model.safetensors.index.json");
  ASSERT_EQ(loader.size(), 1);
  tensor::Tensor tensor;
  ASSERT_TRUE(loader.load("fp32", tensor));
  ASSERT_EQ(tensor.data_type(), base::DataType::kDataTypeFp32);
  ASSERT_FLOAT_EQ(tensor.ptr<float>()[0], 1.5f);
  ASSERT_FLOAT_EQ(tensor.ptr<float>()[1], -2.0f);
}
