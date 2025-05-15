//
// Created by TENDAI on 25-6-9.
//

// 搭配着GetNmsBoxes.cu中的GetNmsBeforeBoxes使用：
// cudaMemsetAsync(GpuOutputCount, 0, 4, stream);
// GetNmsBeforeBoxes(gpu_buffers[1], num_detections, BenignNames.size(), conf_threshold, det_nums, GpuOutputRects, GpuOutputCount, stream);
// cudaMemcpyAsync(CpuOutputCount, GpuOutputCount, sizeof(int), cudaMemcpyDeviceToHost, stream);
// cudaMemcpyAsync(CpuOutputRects, GpuOutputRects, sizeof(DetectRect) * det_nums, cudaMemcpyDeviceToHost, stream);
// cudaStreamSynchronize(stream);
// GetConvDetectionResult(CpuOutputRects, CpuOutputCount, DetectiontRects, input_w, input_h, nms_threshold, result);

struct DetectRect
{
    float classId;
    float score;
    float xmin;
    float ymin;
    float xmax;
    float ymax;
};

void GetConvDetectionResult(DetectRect* OutputRects, int* OutputCount, std::vector<float>& DetectiontRects, int InputW, int InputH, float NmsThresh, std::string& result)
{
    int ret = 0;
    std::vector<DetectRect> detectRects;
    float xmin = 0, ymin = 0, xmax = 0, ymax = 0;

    DetectRect temp;
    for (int i = 0; i < *OutputCount; i++)
    {
        xmin = OutputRects[i].xmin;
        ymin = OutputRects[i].ymin;
        xmax = OutputRects[i].xmax;
        ymax = OutputRects[i].ymax;

        xmin = xmin > 0 ? xmin : 0;
        ymin = ymin > 0 ? ymin : 0;
        xmax = xmax < InputW ? xmax : InputW;
        ymax = ymax < InputH ? ymax : InputH;

        temp.xmin = xmin;
        temp.ymin = ymin;
        temp.xmax = xmax;
        temp.ymax = ymax;
        temp.classId = OutputRects[i].classId;
        temp.score = OutputRects[i].score;
        detectRects.push_back(temp);
    }

    std::sort(detectRects.begin(), detectRects.end(), [](DetectRect& Rect1, DetectRect& Rect2) -> bool
        { return (Rect1.score > Rect2.score); });

    // std::cout << "NMS Before num :" << detectRects.size() << std::endl;
    for (int i = 0; i < detectRects.size(); ++i)
    {
        float xmin1 = detectRects[i].xmin;
        float ymin1 = detectRects[i].ymin;
        float xmax1 = detectRects[i].xmax;
        float ymax1 = detectRects[i].ymax;
        int classId = detectRects[i].classId;
        float score = detectRects[i].score;

        if (classId != -1)
        {
            DetectiontRects.push_back(float(classId));
            DetectiontRects.push_back(float(score));
            DetectiontRects.push_back(float(xmin1));
            DetectiontRects.push_back(float(ymin1));
            DetectiontRects.push_back(float(xmax1));
            DetectiontRects.push_back(float(ymax1));

            for (int j = i + 1; j < detectRects.size(); ++j)
            {
                float xmin2 = detectRects[j].xmin;
                float ymin2 = detectRects[j].ymin;
                float xmax2 = detectRects[j].xmax;
                float ymax2 = detectRects[j].ymax;

                if (detectRects[j].classId == classId)
                {
                    float iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2);
                    if (iou > NmsThresh)
                    {
                        detectRects[j].classId = -1;
                    }
                }
            }
        }
    }
}