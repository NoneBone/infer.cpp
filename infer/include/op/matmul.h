#ifndef INFER_INCLUDE_OP_MATMUL_H_
#define INFER_INCLUDE_OP_MATMUL_H_
#include "op/layer.h"
#include "base/cuda_config.h"
namespace op
{
class MatmulLayer : public LayerParam{
public:
    // 非量化 Qwen3 路径使用 FP16 Tensor；kernel 内部使用 FP32 累加。
    explicit MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1, 
        bool is_quant = false, bool has_bias = false);
    base::Status check() const override;
    base::Status forward() override;
    base::Status set_bias(int32_t idx, int32_t& dims, const void* bias_ptr,
                          base::DeviceType device_type);
    tensor::Tensor& get_bias(int32_t idx);
    const tensor::Tensor& get_bias(int32_t idx) const;

    void to_cuda() override;
private:
    int32_t dim0_ = 0;
    int32_t dim1_ = 0;
    bool has_bias_ = false;
    std::vector<tensor::Tensor> bias_;
};
} // namespace op


#endif
