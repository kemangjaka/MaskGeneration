#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#define CUDA_SAFE_CALL(func) \
do { \
     cudaError_t err = (func); \
     if (err != cudaSuccess) { \
         fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
         exit(err); \
     } \
} while(0)