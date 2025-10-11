cmake_minimum_required(VERSION 3.21)
project(bilinear)

set(CMAKE_CXX_STANDARD 11)
# 设置nvcc编译cu文件时候使用utf-8编码
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /utf-8")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")

enable_language(CUDA)

# 设置cuda多个框架支持
set(CMAKE_CUDA_ARCHITECTURES 75 86)
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

# CUDA
set(CUDA_TOOLKIT_ROOT_DIR C:\\Program\ Files\\NVIDIA\ GPU\ Computing\ Toolkit\\CUDA\\v11.4)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)

add_definitions(-DNOMINMAX)

add_definitions(-DAPI_EXPORTS)

add_executable(bilinear kernel.cu)
include_directories(${CMAKE_SOURCE_DIR}/include)
target_link_libraries(bilinear cudart)

