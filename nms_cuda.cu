//使用方法：
// 其中detection和detection_count是在cuda上的变量，h_detection和detection都是一个float*型变量，六个元素为一组，xywh conf id
    // cudaMemsetAsync(GpuOutputCount, 0, 4, stream);
    // GetNmsBeforeBoxes(gpu_buffers[1], num_detections, BenignNames.size(), conf_threshold, max_det_nums, GpuOutputRects, GpuOutputCount, stream);
    // GetConvDetectionResult_cuda(GpuOutputRects, GpuOutputCount, detection, detection_count, input_w, input_h, nms_threshold);
    // /* cuda nms */
    // if (cuda_nms == true)
    // {
    //     CUDA_CHECK(cudaMemcpyAsync(&h_detection_count, detection_count, sizeof(int), cudaMemcpyDeviceToHost));
    //     if (h_detection_count > 0)
    //     {
    //         h_detection = new float[h_detection_count * 6];
    //         CUDA_CHECK(cudaMemcpyAsync(h_detection, detection, h_detection_count * 6 * sizeof(float), cudaMemcpyDeviceToHost));
    //     }
    //     cudaStreamSynchronize(stream);
    // }


/* cuda nms */
// 设备端IOU计算函数
__device__ float iou_cuda(float xmin1, float ymin1, float xmax1, float ymax1,
    float xmin2, float ymin2, float xmax2, float ymax2) {
    float area1 = (xmax1 - xmin1) * (ymax1 - ymin1);
    float area2 = (xmax2 - xmin2) * (ymax2 - ymin2);

    float inter_xmin = fmaxf(xmin1, xmin2);
    float inter_ymin = fmaxf(ymin1, ymin2);
    float inter_xmax = fminf(xmax1, xmax2);
    float inter_ymax = fminf(ymax1, ymax2);

    float inter_w = fmaxf(inter_xmax - inter_xmin, 0.0f);
    float inter_h = fmaxf(inter_ymax - inter_ymin, 0.0f);
    float inter_area = inter_w * inter_h;

    return inter_area / (area1 + area2 - inter_area);
}

// 边界裁剪核函数
__global__ void ClipBoxes(DetectRect* d_OutputRects, int count, int InputW, int InputH, DetectRect* d_detectRects) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    float xmin = d_OutputRects[idx].xmin;
    float ymin = d_OutputRects[idx].ymin;
    float xmax = d_OutputRects[idx].xmax;
    float ymax = d_OutputRects[idx].ymax;

    xmin = fmaxf(xmin, 0.0f);
    ymin = fmaxf(ymin, 0.0f);
    xmax = fminf(xmax, static_cast<float>(InputW));
    ymax = fminf(ymax, static_cast<float>(InputH));

    d_detectRects[idx] = d_OutputRects[idx];
    d_detectRects[idx].xmin = xmin;
    d_detectRects[idx].ymin = ymin;
    d_detectRects[idx].xmax = xmax;
    d_detectRects[idx].ymax = ymax;
}

// 双调排序核函数（按score降序排序）
__global__ void BitonicSortStep(DetectRect* data, int j, int k, int numElements) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= numElements) return;

    int ixj = i ^ j;
    if (ixj > i) {
        if ((i & k) == 0) {
            // 升序段：降序排序要求交换条件相反
            if (data[i].score < data[ixj].score) {
                DetectRect temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        }
        else {
            // 降序段：降序排序要求交换条件相反
            if (data[i].score > data[ixj].score) {
                DetectRect temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        }
    }
}

// 双调排序封装函数
void BitonicSort(DetectRect* d_data, int numElements) {
    if (numElements == 0) return;

    // 计算最接近的2的幂
    int nextPow2 = 1;
    while (nextPow2 < numElements) nextPow2 <<= 1;

    // 填充数据
    DetectRect* d_padded = nullptr;
    if (nextPow2 != numElements) {
        cudaMalloc(&d_padded, nextPow2 * sizeof(DetectRect));
        cudaMemcpy(d_padded, d_data, numElements * sizeof(DetectRect), cudaMemcpyDeviceToDevice);
        // 填充无效数据（分数为-1）
        cudaMemset(d_padded + numElements, -1, (nextPow2 - numElements) * sizeof(DetectRect));
    }
    else {
        d_padded = d_data;
    }

    // 配置线程
    const int blockSize = 256;
    int gridSize = (nextPow2 + blockSize - 1) / blockSize;

    // 执行双调排序
    for (int k = 2; k <= nextPow2; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            BitonicSortStep << <gridSize, blockSize >> > (d_padded, j, k, nextPow2);
            cudaDeviceSynchronize();
        }
    }

    // 将排序结果复制回原数组
    if (d_padded != d_data) {
        cudaMemcpy(d_data, d_padded, numElements * sizeof(DetectRect), cudaMemcpyDeviceToDevice);
        cudaFree(d_padded);
    }
}

// NMS核函数（顺序执行）
__global__ void NMSKernel(DetectRect* d_detectRects, int count, float* d_DetectiontRects,
    int* d_DetectionCount, float NmsThresh) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int detectionIdx = 0;
    for (int i = 0; i < count; i++) {
        if (d_detectRects[i].classId == -1) continue;

        // 输出当前检测框
        int base = detectionIdx * 6;
        d_DetectiontRects[base] = static_cast<float>(d_detectRects[i].classId);
        d_DetectiontRects[base + 1] = d_detectRects[i].score;
        d_DetectiontRects[base + 2] = d_detectRects[i].xmin;
        d_DetectiontRects[base + 3] = d_detectRects[i].ymin;
        d_DetectiontRects[base + 4] = d_detectRects[i].xmax;
        d_DetectiontRects[base + 5] = d_detectRects[i].ymax;
        detectionIdx++;

        // 抑制重叠框
        for (int j = i + 1; j < count; j++) {
            if (d_detectRects[j].classId != d_detectRects[i].classId) continue;

            float iou = iou_cuda(d_detectRects[i].xmin, d_detectRects[i].ymin,
                d_detectRects[i].xmax, d_detectRects[i].ymax,
                d_detectRects[j].xmin, d_detectRects[j].ymin,
                d_detectRects[j].xmax, d_detectRects[j].ymax);

            if (iou > NmsThresh) {
                d_detectRects[j].classId = -1; // 标记为抑制
            }
        }
    }

    *d_DetectionCount = detectionIdx;
}

// 主函数：获取检测结果
void GetConvDetectionResult_cuda(DetectRect* d_OutputRects, int* d_OutputCount,
    float* d_DetectiontRects, int* d_DetectionCount,
    int InputW, int InputH, float NmsThresh) {
    int count;
    cudaMemcpy(&count, d_OutputCount, sizeof(int), cudaMemcpyDeviceToHost);
    if (count <= 0) {
        cudaMemset(d_DetectionCount, 0, sizeof(int));
        return;
    }

    // 分配设备内存
    DetectRect* d_detectRects;
    cudaMalloc(&d_detectRects, count * sizeof(DetectRect));

    // 步骤1：边界裁剪
    const int blockSize = 256;
    int gridSize = (count + blockSize - 1) / blockSize;
    ClipBoxes << <gridSize, blockSize >> > (d_OutputRects, count, InputW, InputH, d_detectRects);
    cudaDeviceSynchronize();

    // 步骤2：双调排序（按分数降序）
    BitonicSort(d_detectRects, count);

    // 步骤3：执行NMS
    NMSKernel << <1, 1 >> > (d_detectRects, count, d_DetectiontRects, d_DetectionCount, NmsThresh);
    cudaDeviceSynchronize();

    // 清理设备内存
    cudaFree(d_detectRects);
}

