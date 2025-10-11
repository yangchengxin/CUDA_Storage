#include <cuda_runtime.h>
#include <iostream>
 
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
 
using namespace std;
typedef unsigned char uchar;


// cpu 读取图像文件
uchar3* read_image(const char* filename ){
    
    int width, height, channels;
    // 读取图像文件
    unsigned char* imageData = stbi_load(filename, &width, &height, &channels, 0);
    if (imageData == nullptr) {
        std::cerr << "Error: Could not load image " << filename << std::endl;
    }
 
    std::cout << "Image loaded: " << filename << std::endl;
    std::cout << "Width: " << width << " Height: " << height << " Channels: " << channels << std::endl;
    
    int numPixels = width * height;
    uchar3* image = new uchar3[numPixels];
    for (int i = 0; i < numPixels; ++i) {
        image[i].x = imageData[i * 3 + 0];  // R
        image[i].y = imageData[i * 3 + 1];  // G
        image[i].z = imageData[i * 3 + 2];  // B
    }
    return image;
}

// 计算每个像素rgb的插值结果
__device__ uchar3  bilinearInterpolation_test(float srcX, float srcY, 
                                        uchar3* d_inputImage, int inputWidth, int inputHeight,
                                        uchar3& res){
    // 找到周围的四个像素
    int x1 = (int)floor(srcX);
    int y1 = (int)floor(srcY);
    int x2 = min(x1 + 1, inputWidth - 1);
    int y2 = min(y1 + 1, inputHeight - 1);
 
    // 计算插值权重
    float wx = srcX - x1;
    float wy = srcY - y1;
 
    // 双线性插值计算（相邻四个点的像素值）
    uchar3 p1 = d_inputImage[y1 * inputWidth + x1];
    uchar3 p2 = d_inputImage[y1 * inputWidth + x2];
    uchar3 p3 = d_inputImage[y2 * inputWidth + x1];
    uchar3 p4 = d_inputImage[y2 * inputWidth + x2];
 
    uchar3 interpolated;
    // 插值计算
    interpolated.x = (uchar)((1 - wx) * (1 - wy) * p1.x + wx * (1 - wy) * p2.x + (1 - wx) * wy * p3.x + wx * wy * p4.x);
    interpolated.y = (uchar)((1 - wx) * (1 - wy) * p1.y + wx * (1 - wy) * p2.y + (1 - wx) * wy * p3.y + wx * wy * p4.y);
    interpolated.z = (uchar)((1 - wx) * (1 - wy) * p1.z + wx * (1 - wy) * p2.z + (1 - wx) * wy * p3.z + wx * wy * p4.z);
    return interpolated;
}
 
__global__ void bilinearInterpolationKernel(
                                  uchar3* d_inputImage, 
                                  uchar3* d_outputImage, 
                                  int inputWidth, int inputHeight, 
                                  int outputWidth, int outputHeight,
                                  float scaleX, float scaleY
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if( x < outputWidth && y < outputHeight ){
        // 计算在源图像中位置
        float srcX = x * scaleX;
        float srcY = y * scaleY;
        
        uchar3 interpolated_tmp;
        uchar3 interpolated_tmp2 = bilinearInterpolation_test(  srcX, srcY,
                                d_inputImage,  inputWidth,   inputHeight,
                                interpolated_tmp);
        d_outputImage[(y *outputWidth + x )] = interpolated_tmp2;
    }
}

void bilinearInterpolation_launch(uchar3*  h_inputImageUChar3, 
                                  uchar3*  h_outputImageUChar3, 
                                  int inputWidth, int inputHeight, 
                                  int outputWidth, int outputHeight){
    uchar3* d_inputImage;
    uchar3* d_outputImage;
 
    size_t inputImageSize = inputWidth * inputHeight * sizeof(uchar3);
    size_t outputImageSize = outputWidth * outputHeight * sizeof(uchar3);
    cout << "sizeof(uchar3) = " << sizeof(uchar3) << endl;
 
    // cuda malloc && memset
    cudaMalloc(&d_inputImage, inputImageSize);
    cudaMalloc(&d_outputImage, outputImageSize);
    cudaMemset(d_inputImage, 0, inputImageSize);
    cudaMemset(d_outputImage, 0, outputImageSize);
 
    // h2d
    auto status = cudaMemcpy( d_inputImage, h_inputImageUChar3, inputImageSize, cudaMemcpyHostToDevice );
    cout << "h2d status = " << status << endl;
 
    float scaleX = (float)(inputWidth -  1) / outputWidth;
    float scaleY = (float)(inputHeight - 1) / outputHeight;
 
    // cuda block/grid size
    dim3 blockSize(16,16,1);
    dim3 gridSize( (outputWidth + blockSize.x -1) /blockSize.x, \
                     (outputHeight + blockSize.y -1) /blockSize.y,1  );
    cout << "blockSize: x =" << blockSize.x <<",y = " << blockSize.y <<",z ="<< blockSize.z << endl;
    cout << "gridSize: x = " << gridSize.x <<",y="<< gridSize.y <<",z = "<< gridSize.z<< endl;
 
    // 双线性插值算法
    bilinearInterpolationKernel<<<gridSize,blockSize >>>(d_inputImage,d_outputImage,inputWidth, inputHeight,outputWidth, outputHeight,scaleX,scaleY );
 
 
    // 同步设备
    cudaDeviceSynchronize();
 
    // 复制输出图像数据回主机
    cudaMemcpy(h_outputImageUChar3, d_outputImage, outputImageSize, cudaMemcpyDeviceToHost);
 
    // 释放设备内存
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}

int main()
{
 
    int inputWidth   = 640;
    int inputHeight  = 427;
    int outputWidth  = 320;
    int outputHeight = 213;
 
    // 读取图片
    const char* image_path = "D:\\ycx_git_repositories\\CUDA_Storage\\bilinear\\cat.jpg";
    uchar3* h_inputImage = read_image(image_path);
 
    // malloc host 
    uchar3* h_outputImage = new uchar3[outputWidth * outputHeight * 3];
    
    // 调用cuda launch函数
    bilinearInterpolation_launch(h_inputImage, h_outputImage, inputWidth, inputHeight, outputWidth, outputHeight);
 
    // save img 
    const char* output_filename = "D:\\ycx_git_repositories\\CUDA_Storage\\bilinear\\cat_1.jpg";
    stbi_write_png( output_filename, outputWidth, outputHeight, 3, h_outputImage, outputWidth * 3);
    
    // free cpu 
    delete[] h_inputImage;
    delete[] h_outputImage;
 
    return 0;
}