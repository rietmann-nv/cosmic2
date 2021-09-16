#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <cuda_runtime.h>

void descramble_cuda(void * frame8, size_t rows);
void descramble_cpu(void * frame8, size_t rows);

#define CUDA_CHECK( call )                                          \
  {                                                                 \
    cudaError_t rc = call;                                          \
    if (rc != cudaSuccess) {                                        \
      fprintf(stderr,                                               \
              "CUDA call (%s) failed with code %d (line %d): %s\n", \
              #call, rc, __LINE__, cudaGetErrorString(rc));         \
      throw std::runtime_error("fatal cuda error");                 \
    }                                                               \
  }


int main() {

  int nrows = 10;
  unsigned short* data = (unsigned short*)malloc(192*nrows*sizeof(unsigned short));
  unsigned short* data2 = (unsigned short*)malloc(192*nrows*sizeof(unsigned short));
  for(int i=0; i<nrows*192; i++) {
    data[i] = rand();
    data2[i] = data[i];
  }

  descramble_cpu(data, nrows);

  using uint16_t = unsigned short;
  uint16_t * d_frame;
  CUDA_CHECK(cudaMalloc(&d_frame, 192*nrows*sizeof(uint16_t)));
  CUDA_CHECK(cudaMemcpy(d_frame, data2, 192*nrows*sizeof(uint16_t), cudaMemcpyHostToDevice));
  descramble_cuda(d_frame, nrows);
  CUDA_CHECK(cudaMemcpy(data2, d_frame, 192*nrows*sizeof(uint16_t), cudaMemcpyDeviceToHost));

  for(int i=0; i<nrows*192; i++) {
    if(data[i] != data2[i]) {
      printf("ERROR[%d] %d != %d\n", i, data[i], data2[i]);
      cudaFree(d_frame);
      return 1;
    }
  }
  cudaFree(d_frame);
  return 0;
}
