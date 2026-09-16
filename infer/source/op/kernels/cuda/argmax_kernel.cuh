#ifndef ARGMAX_KERNEL_CUH
#define ARGMAX_KERNEL_CUH
#include <cuda_fp16.h>
namespace kernel {
size_t argmax_kernel_cu(const __half* input_ptr, size_t size, void* stream);
}
#endif  // ARGMAX_KERNEL_CUH
