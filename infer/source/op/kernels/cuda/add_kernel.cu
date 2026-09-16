#include "add_kernel.cuh"


namespace kernel{
__global__ void add_kernel_cu_fp16(const __half* a, const __half* b, __half* c, int32_t size){
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size){return;}
    c[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]));
}

void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                const tensor::Tensor& output, void* stream){
    CHECK_EQ(input1.is_empty(), false);
    CHECK_EQ(input2.is_empty(), false);
    CHECK_EQ(output.is_empty(), false);

    CHECK_EQ(input1.size(), input2.size());
    CHECK_EQ(input1.size(), output.size());
    CHECK(input1.data_type() == base::DataType::kDataTypeFp16);
    CHECK(input2.data_type() == base::DataType::kDataTypeFp16);
    CHECK(output.data_type() == base::DataType::kDataTypeFp16);

    int32_t size = input1.size();
    int32_t thread = 512;
    int32_t block = CEIL(size, thread);

    if(stream){
        cudaStream_t stream_ = static_cast<CUstream_st*>(stream);// issue
        add_kernel_cu_fp16<<<block, thread, 0, stream_>>>(input1.ptr<__half>(),
        input2.ptr<__half>(), const_cast<__half*>(output.ptr<__half>()), size);
    }else{
        add_kernel_cu_fp16<<<block, thread>>>(input1.ptr<__half>(),
        input2.ptr<__half>(), const_cast<__half*>(output.ptr<__half>()), size);
    }
};
} // namespace kernel
