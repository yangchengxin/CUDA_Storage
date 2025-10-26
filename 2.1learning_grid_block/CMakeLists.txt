cmake_minimum_required(VERSION 3.21)
project(grid_block)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /utf-8")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")

enable_language(CUDA)

set(CMAKE_CUDA_ARCHITECTURES 52 75 86)
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

# CUDA
set(CUDA_TOOLKIT_ROOT_DIR C:\\Program\ Files\\NVIDIA\ GPU\ Computing\ Toolkit\\CUDA\\v12.4)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)

add_definitions(-DNOMINMAX)
add_definitions(-DAPI_EXPORTS)

add_executable(${PROJECT_NAME} print_dim.cu)
include_directories(${CMAKE_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} cudart)