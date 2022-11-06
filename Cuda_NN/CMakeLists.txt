cmake_minimum_required (VERSION 3.0)
project (myproject)
enable_language(CUDA)
find_package(CUDA REQUIRED)	#find the CUDA libraries
include_directories(${CUDA_INCLUDE_DIRS})	#Specify the CUDA include directories for the project

include_directories ("${PROJECT_SOURCE_DIR}/third_party/eigen")
# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_SRCS 变量
aux_source_directory(. DIR_SRCS)
# 添加 math 子目录
add_subdirectory(src)
# 指定生成目标 
cuda_add_executable(demo demo.cu)
cuda_add_executable(test time_inference.cu)
# 添加链接库
target_link_libraries(demo MiniDNNLib)
target_link_libraries(test MiniDNNLib)

