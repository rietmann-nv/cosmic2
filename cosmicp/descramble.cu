
#include <stdio.h>
#include <stdexcept>
// using uint16_t = unsigned short;
typedef unsigned short uint16_t;





__global__ void descramble_kernel(uint16_t* frame, int* des, size_t rows) {

  const int n = threadIdx.x;
  const int off = blockIdx.x;

  __shared__ uint16_t row[192];

  row[n] = frame[n+off*192];
  __syncthreads();

  frame[n+off*192] = row[des[n]];

}

void descramble_cuda(void * frame8, size_t rows);
void descramble_cpu(void * frame8, size_t rows);

void descramble_cuda(void * d_frame8, size_t rows) {
  
  printf("Hello from cuda\n");
  uint16_t * d_frame = (uint16_t * )frame8;
  

  const int des[192] = { 
    188, 172, 156, 140, 124, 108, 92, 76, 60, 44, 28, 12, 
    189, 173, 157, 141, 125, 109, 93, 77, 61, 45, 29, 13, 
    190, 174, 158, 142, 126, 110, 94, 78, 62, 46, 30, 14, 
    191, 175, 159, 143, 127, 111, 95, 79, 63, 47, 31, 15, 
    184, 168, 152, 136, 120, 104, 88, 72, 56, 40, 24, 8, 
    185, 169, 153, 137, 121, 105, 89, 73, 57, 41, 25, 9, 
    186, 170, 154, 138, 122, 106, 90, 74, 58, 42, 26, 10, 
    187, 171, 155, 139, 123, 107, 91, 75, 59, 43, 27, 11, 
    180, 164, 148, 132, 116, 100, 84, 68, 52, 36, 20, 4, 
    181, 165, 149, 133, 117, 101, 85, 69, 53, 37, 21, 5, 
    182, 166, 150, 134, 118, 102, 86, 70, 54, 38, 22, 6, 
    183, 167, 151, 135, 119, 103, 87, 71, 55, 39, 23, 7, 
    176, 160, 144, 128, 112, 96, 80, 64, 48, 32, 16, 0, 
    177, 161, 145, 129, 113, 97, 81, 65, 49, 33, 17, 1, 
    178, 162, 146, 130, 114, 98, 82, 66, 50, 34, 18, 2, 
    179, 163, 147, 131, 115, 99, 83, 67, 51, 35, 19, 3};

  int* d_des;
  CUDA_CHECK(cudaMalloc(&d_des, sizeof(int)*192));
  CUDA_CHECK(cudaMemcpy(d_des, des, sizeof(int)*192, cudaMemcpyHostToDevice));

  const int num_blocks = rows;
  const int num_threads = 192;

  descramble_kernel<<<num_blocks, num_threads>>>(d_frame, d_des, rows);
  CUDA_CHECK(cudaMemcpy(frame8, d_frame, 192*rows*sizeof(uint16_t), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();
}

void descramble_cpu(void *frame8, size_t rows){
 const int des[192] = { 
    188, 172, 156, 140, 124, 108, 92, 76, 60, 44, 28, 12, 
    189, 173, 157, 141, 125, 109, 93, 77, 61, 45, 29, 13, 
    190, 174, 158, 142, 126, 110, 94, 78, 62, 46, 30, 14, 
    191, 175, 159, 143, 127, 111, 95, 79, 63, 47, 31, 15, 
    184, 168, 152, 136, 120, 104, 88, 72, 56, 40, 24, 8, 
    185, 169, 153, 137, 121, 105, 89, 73, 57, 41, 25, 9, 
    186, 170, 154, 138, 122, 106, 90, 74, 58, 42, 26, 10, 
    187, 171, 155, 139, 123, 107, 91, 75, 59, 43, 27, 11, 
    180, 164, 148, 132, 116, 100, 84, 68, 52, 36, 20, 4, 
    181, 165, 149, 133, 117, 101, 85, 69, 53, 37, 21, 5, 
    182, 166, 150, 134, 118, 102, 86, 70, 54, 38, 22, 6, 
    183, 167, 151, 135, 119, 103, 87, 71, 55, 39, 23, 7, 
    176, 160, 144, 128, 112, 96, 80, 64, 48, 32, 16, 0, 
    177, 161, 145, 129, 113, 97, 81, 65, 49, 33, 17, 1, 
    178, 162, 146, 130, 114, 98, 82, 66, 50, 34, 18, 2, 
    179, 163, 147, 131, 115, 99, 83, 67, 51, 35, 19, 3};

  uint16_t * frame = (uint16_t *)frame8;
  //uint16_t * restrict from = (uint16_t * restrict)src;
  uint16_t row[192]; // = malloc(packet_size)
  for (size_t off=0;off < rows;off++){
    for (size_t n =0;n<192;n++){
        *(row+n) = *(frame+n+off*192) ;
    }
    for (size_t n =0;n<192;n++){
      *(frame+n+off*192) = *(row+des[n]);
    }
    //rest
    //for (int n =0;n<sze % 192;n++){
    //    *(dest+n+off) = *(src+des[n]+off) ;
    //}
  }
}
