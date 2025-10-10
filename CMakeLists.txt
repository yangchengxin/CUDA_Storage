cmake_minimum_required(VERSION 3.21)
project(yolo11_trtx_v8)

set(CMAKE_CXX_STANDARD 11)
# 设置nvcc编译cu文件时候使用utf-8编码
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /utf-8")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")

enable_language(CUDA)

# 设置cuda多个框架支持
set(CMAKE_CUDA_ARCHITECTURES 75 86)
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

# # OpenCV
# set(OpenCV_DIR D:\\opencv\\build\\x64\\vc16\\lib)
# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})
# link_directories(${OpenCV_LIB_DIR})

# CUDA
set(CUDA_TOOLKIT_ROOT_DIR C:\\Program\ Files\\NVIDIA\ GPU\ Computing\ Toolkit\\CUDA\\v11.4)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)

add_definitions(-DNOMINMAX)

add_definitions(-DAPI_EXPORTS)

add_executable(hello_cuda hello.cu)
target_link_libraries(hello_cuda cudart)

