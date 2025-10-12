# CUDA_Storage

## 📌 代码列表

| **代码名称** | **介绍** |
|-------------|---------|
| [GetNmsBoxes.cu](https://github.com/yangchengxin/CUDA_Storage/blob/main/GetNmsBoxes.cu) | 从所有grid的结果中筛选出满足符合置信度要求的结果送到nms中。 |
| [gamma.cu](https://github.com/yangchengxin/CUDA_Storage/blob/main/gamma.cu) | 对图像进行gamma增强，并将增强后的结果和原图拼成一个三通道图像。 |
| [letterbox.cu](https://github.com/yangchengxin/CUDA_Storage/blob/main/letterbox.cu) | yolo等算法中常用的保持输入长宽比且满足网络输入尺寸的方法。 |
| [nms_cpu](https://github.com/yangchengxin/CUDA_Storage/blob/main/nms_cpu.cpp) | 在cpu上进行nms。 |
| [nms_cuda](https://github.com/yangchengxin/CUDA_Storage/blob/main/nms_cuda.cu) | 在cuda上进行nms。 |
| [biliearInterpolation](https://github.com/yangchengxin/CUDA_Storage/tree/main/bilinear) | 二次线性插值resize |
| [2.1.1learning_grid_block](https://github.com/yangchengxin/CUDA_Storage/tree/main/2.1.1learning_grid_block) | grid，block与thread的索引关系 |
| [2.1.2cu_cpp_interactive](https://github.com/yangchengxin/CUDA_Storage/tree/main/2.1.2cu_cpp_interactive) | cpp与cuda间的交互 |


## 📌 使用方法
1. **克隆仓库**：
   ```sh
   git clone https://github.com/yangchengxin/CUDA_Storage.git
