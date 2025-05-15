#include "postprocess.h"
#include "cuda_utils.h"
#include "device_launch_parameters.h"
#include <algorithm>
#include <cmath>


__global__ void GetNmsBeforeBoxesKernel(float* SrcInput, int AnchorCount, int ClassNum, float ObjectThresh, int NmsBeforeMaxNum, DetectRect* OutputRects, int* OutputCount)
{
    /***
    功能说明：用8400个线程，实现对80个类别选出最大值，并判断是否大于阈值，把大于阈值的框记录下来后面用于参加mns
    SrcInput: 模型输出（1,84,8400）
    AnchorCount: 8400
    ClassNum: 80
    ObjectThresh: 目标阈值（大于该阈值的目标才输出）
    NmsBeforeMaxNum: 输入nms检测框的最大数量，前面申请的了一块儿显存来装要参加nms的框，防止越界
    OutputRects: 大于阈值的目标框
    OutputCount: 大于阈值的目标框个数
    ***/

    int ThreadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (ThreadId >= AnchorCount)
    {
        return;
    }

    float* XywhConf = SrcInput + ThreadId;
    float CenterX = 0, CenterY = 0, CenterW = 0, CenterH = 0;

    float MaxScore = 0;
    int MaxIndex = 0;

    DetectRect TempRect;
    for (int j = 4; j < ClassNum + 4; j++)
    {
        if (4 == j)
        {
            MaxScore = XywhConf[j * AnchorCount];
            MaxIndex = j;
        }
        else
        {
            if (MaxScore < XywhConf[j * AnchorCount])
            {
                MaxScore = XywhConf[j * AnchorCount];
                MaxIndex = j;
            }
        }
    }

    if (MaxScore > ObjectThresh)
    {
        int index = atomicAdd(OutputCount, 1);

        if (index > NmsBeforeMaxNum)
        {
            return;
        }

        CenterX = XywhConf[0 * AnchorCount];
        CenterY = XywhConf[1 * AnchorCount];
        CenterW = XywhConf[2 * AnchorCount];
        CenterH = XywhConf[3 * AnchorCount];

        TempRect.classId = MaxIndex - 4;
        TempRect.score = MaxScore;
        TempRect.xmin = CenterX - 0.5 * CenterW;
        TempRect.ymin = CenterY - 0.5 * CenterH;
        TempRect.xmax = CenterX + 0.5 * CenterW;
        TempRect.ymax = CenterY + 0.5 * CenterH;

        OutputRects[index] = TempRect;
    }
}


void GetNmsBeforeBoxes(float* SrcInput, int AnchorCount, int ClassNum, float ObjectThresh, int NmsBeforeMaxNum, DetectRect* OutputRects, int* OutputCount, cudaStream_t Stream)
{
    int Block = 512;
    int Grid = (AnchorCount + Block - 1) / Block;

    GetNmsBeforeBoxesKernel << <Grid, Block, 0, Stream >> > (SrcInput, AnchorCount, ClassNum, ObjectThresh, NmsBeforeMaxNum, OutputRects, OutputCount);
    return;
}