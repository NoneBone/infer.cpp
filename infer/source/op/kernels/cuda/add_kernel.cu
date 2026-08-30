#include "add_kernel.cuh"


namespace kernel{
__global__ void add_kernel_cu_fp32(const float* a, const float* b, float* c, int32_t size){
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size){return;}
    c[idx] = a[idx] + b[idx];
}

void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                const tensor::Tensor& output, void* stream){
    CHECK_EQ(input1.is_empty(), false);
    CHECK_EQ(input2.is_empty(), false);
    CHECK_EQ(output.is_empty(), false);

    CHECK_EQ(input1.size(), input2.size());
    CHECK_EQ(input1.size(), output.size());

    int32_t size = input1.size();
    int32_t thread = 512;
    int32_t block = CEIL(size, thread);

    if(stream){
        cudaStream_t stream_ = static_cast<CUstream_st*>(stream);// issue
        add_kernel_cu_fp32<<<thread, block, 0, stream_>>>(input1.ptr<float>(), 
        input2.ptr<float>(), const_cast<float*>(output.ptr<float>()), size);
    }else{
        add_kernel_cu_fp32<<<thread, block>>>(input1.ptr<float>(), 
        input2.ptr<float>(), const_cast<float*>(output.ptr<float>()), size);
    }
};
} // namespace kernel

