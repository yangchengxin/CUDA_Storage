#include "cuda_runtime.h"
#include "cuda.h"
#include "stdio.h"
#include "matmul.hpp"

__global__ void MatmulKernel(float *M_device, float *N_device, float *P_device, int width){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float P_element = 0;

    for (int k = 0; k < width; k ++){
        float M_element = M_device[y * width + k];
        float N_element = N_device[k * width + x];
        P_element += M_element * N_element;
    }

    P_device[y * width + x] = P_element;
}

void MatmulOnDevice(float *M_host, float *N_host, float* P_host, int width, int blockSize){
    int size = width * width * sizeof(float);

    float *M_device;
    float *N_device;

    cudaMalloc(&M_device, size);
    cudaMalloc(&N_device, size);

    cudaMemcpy(M_device, M_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_device, N_host, size, cudaMemcpyHostToDevice);

    float *P_device;
    cudaMalloc(&P_device, size);

    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid(width / blockSize, width / blockSize);
    MatmulKernel <<<dimGrid, dimBlock>>> (M_device, N_device, P_device, width);

    cudaMemcpy(P_host, P_device, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    /* Free */
    cudaFree(P_device);
    cudaFree(N_device);
    cudaFree(M_device);
}

