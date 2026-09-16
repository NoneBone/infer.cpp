#ifndef TEST_CU_CUH
#define TEST_CU_CUH
#include <cuda_fp16.h>

void set_value_cu(float* dst, int size, float val = 1.0f);
void set_value_cu_fp16(__half* dst, int size, float val = 1.0f);
#define CEIL(a,b) ((a+b-1)/b)
#endif