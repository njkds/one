cmake_minimum_required(VERSION 2.6)
project(v1)
add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)
set(HAS_PYTHON ON)

set(CMAKE_CXX_STANDARD 14)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")

# 如果要支持python则设置python路径
set(PythonRoot "/home/dell/anaconda3/envs/yolov5")
set(PythonName "python3.8")

# 如果你是不同显卡，请设置为显卡对应的号码参考这里：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
set(CUDA_GEN_CODE "-gencode=arch=compute_75,code=sm_75")

# 如果你的opencv找不到，可以自己指定目录
set(OpenCV_DIR   "/home/dell/opencv4/lib/cmake/opencv4")
#set(OpenCV_DIR "~/opencv/opencv-4.5.4/build")

# 建议cuda10.2以上
set(CUDA_DIR     "/usr/local/cuda")

# 需要与cuda、tensorRT配对
set(CUDNN_DIR    "/usr/local/cuda")

# 注意支持8.0以上，以下不见得支持
set(TENSORRT_DIR "/home/dell/TensorRT-8.2.1.8")

# 因为protobuf，需要用特定版本，所以这里指定路径
set(PROTOBUF_DIR "/usr/local")

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(k4a REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/application
    ${PROJECT_SOURCE_DIR}/src/tensorRT
    ${PROJECT_SOURCE_DIR}/src/tensorRT/common
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_DIR}/include
    ${PROTOBUF_DIR}/include
    ${TENSORRT_DIR}/include
    ${CUDNN_DIR}/include
    ${NUMCPP_DIR}/include
)

# 切记，protobuf的lib目录一定要比tensorRT目录前面，因为tensorRTlib下带有protobuf的so文件
# 这可能带来错误
link_directories(
    ${PROTOBUF_DIR}/lib
    ${TENSORRT_DIR}/lib
    ${CUDA_DIR}/lib64
    ${CUDNN_DIR}/lib
)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O0 -Wfatal-errors -pthread -w -g")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")
file(GLOB_RECURSE cpp_srcs ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE cuda_srcs ${PROJECT_SOURCE_DIR}/src/*.cu)
cuda_add_library(plugin_list SHARED ${cuda_srcs})

add_executable(v1 ${cpp_srcs})
target_link_libraries(v1 nvinfer nvinfer_plugin)
target_link_libraries(v1 cuda cublas cudart cudnn)
target_link_libraries(v1 protobuf pthread plugin_list)
target_link_libraries(v1 ${OpenCV_LIBS})
target_link_libraries(v1 k4a::k4a)
target_link_libraries(v1 ${PCL_LIBRARIES})

add_custom_target(
        yolo
        DEPENDS v1
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
        COMMAND ./v1 yolo
)