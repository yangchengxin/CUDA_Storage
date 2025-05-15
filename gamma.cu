#include "preprocess.h"
#include "cuda_utils.h"
#include "device_launch_parameters.h"
#include <algorithm>
#include <cmath>

/* ------ 单通道gamma校正 ------ */
__global__ void gammaCorrectionKernel(const uint8_t* src, uint8_t* dst, int width, int height, float gamma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        float pixel = static_cast<float>(src[idx]) / 255.0f;
        pixel = powf(pixel, gamma) * 255.0f;
        dst[idx] = min(255, max(0, static_cast<int>(pixel)));
    }
}

/* ------ 三通道bgr变成单通道灰度图，并gamma校正 ------ */
__global__ void convertToGrayKernel(const uchar3* src, uint8_t* dst, int width, int height, float gamma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        uchar3 pixel = src[idx];
        // bgr
        dst[idx] = static_cast<uint8_t>(
            0.114f * pixel.x +      // b通道权重 0.114
            0.587f * pixel.y +      // g通道权重 0.587
            0.299f * pixel.z        // r通道权重 0.299
        );

        float pixel_f = static_cast<float>(dst[idx]) / 255.0f;
        pixel_f = powf(pixel_f, gamma) * 255.0f;
        dst[idx] = min(255, max(0, static_cast<int>(pixel_f)));
    }
}

/* ------ 通道合并 ------ */
__global__ void mergeChannelsKernel(const uint8_t* gray1, const uint8_t* gray2, const uint8_t* gray3, uchar3* dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        uchar3 pixel;
        pixel.x = gray1[idx];
        pixel.y = gray2[idx];
        pixel.z = gray3[idx];
        dst[idx] = pixel;
    }
}

/* ------------------ CUDA包装函数 ----------------- */
void cuda_gamma_merge(cv::Mat& src, cv::Mat& dst) {
    int width = src.cols, height = src.rows;

    uchar3* d_src;
    uint8_t* d_gray1, * d_gray2, * d_gray3;
    uchar3* d_dst;

    size_t imgSize = width * height * sizeof(uchar3);
    size_t graySize = width * height * sizeof(uint8_t);

    cudaMalloc(&d_src, imgSize);
    cudaMalloc(&d_gray1, graySize);
    cudaMalloc(&d_gray2, graySize);
    cudaMalloc(&d_gray3, graySize);
    cudaMalloc(&d_dst, imgSize);

    cudaMemcpy(d_src, src.data, imgSize, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // 将输入的bgr原图转成单通道灰度图
    convertToGrayKernel << <grid, block >> > (d_src, d_gray2, width, height, 1.0);
    // 将输入的bgr原图 ---> 1.5 gamma变换 ---> 单通道灰度图
    gammaCorrectionKernel << <grid, block >> > (d_gray2, d_gray1, width, height, 1.5);
    // 将输入的bgr原图 ---> 0.8 gamma变换 ---> 单通道灰度图
    gammaCorrectionKernel << <grid, block >> > (d_gray2, d_gray3, width, height, 0.8);


    // 1.5, 1, 0.8的顺序
    mergeChannelsKernel << <grid, block >> > (d_gray1, d_gray2, d_gray3, d_dst, width, height);

    cudaMemcpy(dst.data, d_dst, imgSize, cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_gray1);
    cudaFree(d_gray2);
    cudaFree(d_gray3);
    cudaFree(d_dst);
}

void cuda_preprocess(
    uint8_t* src, int src_width, int src_height,
    float* dst, int dst_width, int dst_height,
    cudaStream_t stream) {

    // 计算图像存储的字节大小
    int img_size = src_width * src_height * 3;
    // 将图像数据拷贝到分页内存上
    memcpy(img_buffer_host, src, img_size);
    // 将分页内存的图像拷贝到cuda上
    CUDA_CHECK(cudaMemcpyAsync(img_buffer_device, img_buffer_host, img_size, cudaMemcpyHostToDevice, stream));

    AffineMatrix s2d, d2s;
    float scale = std::min(dst_height / (float)src_height, dst_width / (float)src_width);

    s2d.value[0] = scale;
    s2d.value[1] = 0;
    s2d.value[2] = -scale * src_width * 0.5 + dst_width * 0.5;
    s2d.value[3] = 0;
    s2d.value[4] = scale;
    s2d.value[5] = -scale * src_height * 0.5 + dst_height * 0.5;

    cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
    cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
    cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);

    memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

    int jobs = dst_height * dst_width;
    int threads = 256;
    int blocks = ceil(jobs / (float)threads);

    warpaffine_kernel << <blocks, threads, 0, stream >> > (
        img_buffer_device, src_width * 3, src_width,
        src_height, dst, dst_width,
        dst_height, 114, d2s, jobs);
}