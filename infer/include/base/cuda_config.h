#ifndef BLAS_HELPER_H
#define BLAS_HELPER_H
#include <cuda_runtime_api.h>
#include <cublas.h>
namespace kernel{
struct CudaConfig{
    cudaStream_t stream = nullptr;
    ~CudaConfig(){
        if(stream){
            cudaStreamDestroy(stream);
        }
    }
};
} // namespace kernel
#endif  // BLAS_HELPER_H
