cmake_minimum_required(VERSION 3.0.0)
project(MaskGeneration VERSION 0.1.0)

set(CMAKE_BUILD_TYPE Release)

#add_executable(MaskGeneration Source.cpp Header.h)
file(GLOB mainEngine mainEngine/*.cpp)
file(GLOB DatasetReader DatasetReader/*.cpp)
file(GLOB DimensionConvertor DimensionConvertor/*.cpp DimensionConvertor/*.cu)
file(GLOB GeometricSegmentation GeometricSegmentation/*.cpp GeometricSegmentation/*.cu)
file(GLOB JointBilateralFilter JointBilateralFilter/*.cpp JointBilateralFilter/*.cu)
file(GLOB NormalMapGenerator NormalMapGenerator/*.cpp NormalMapGenerator/*.cu)
file(GLOB ObjectDetector ObjectDetector/*.cpp)
file(GLOB RGBDSceneReader RGBDSceneReader/*.cpp)
file(GLOB NYUv2Reader NYUv2Reader/*.cpp)
file(GLOB CoFusionReader CoFusionReader/*.cpp)
file(GLOB photometricSegmentation photometricSegmentation/*.cpp)
file(GLOB utils Utils/*.cpp)

#add_executable(MaskGeneration Source.cpp Header.h)
file(GLOB header_mainEngine mainEngine/*.h)
file(GLOB header_DatasetReader DatasetReader/*.h)
file(GLOB header_DimensionConvertor DimensionConvertor/*.h)
file(GLOB header_GeometricSegmentation GeometricSegmentation/*.h)
file(GLOB header_JointBilateralFilter JointBilateralFilter/*.h)
file(GLOB header_NormalMapGenerator NormalMapGenerator/*.h)
file(GLOB header_ObjectDetector ObjectDetector/*.h)
file(GLOB header_RGBDSceneReader RGBDSceneReader/*.h)
file(GLOB header_CoFusionReader CoFusionReader/*.h)
file(GLOB header_photometricSegmentation photometricSegmentation/*.h)
file(GLOB header_utils Utils/*.h)

include_directories(header_mainEngine)
include_directories(header_DatasetReader)
include_directories(header_DimensionConvertor)
include_directories(header_GeometricSegmentation)
include_directories(header_JointBilateralFilter)
include_directories(header_NormalMapGenerator)
include_directories(header_ObjectDetector)
include_directories(header_RGBDSceneReader)
include_directories(header_CoFusionReader)
include_directories(header_NYUv2Reader)
include_directories(header_photometricSegmentation)
include_directories(header_utils)

# For OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    message(${OpenCV_INCLUDE_DIRS})
endif()

# For multi-threading
find_package (Threads REQUIRED)

option(WITH_OPENMP "Compile with OpenMP" ON)
if(WITH_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    message(STATUS "Compiling with OpenMP support")
  endif()
endif()
if (WITH_OPENMP AND OPENMP_FOUND)
    list(APPEND compile_flags ${OpenMP_CXX_FLAGS})
    list(APPEND libraries ${OpenMP_CXX_FLAGS})
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_compile_definitions(_OPENMP)
endif()

#For CUDA
FIND_PACKAGE(CUDA REQUIRED)
INCLUDE(FindCUDA)
INCLUDE_DIRECTORIES(/usr/local/cuda/include)

LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg "-std=c++11")
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)
set(CUDART_LIBS /usr/local/cuda/lib64/libcudart.so)


#For YOLO
include_directories(/home/ryo/Lib/darknet/include)
set(DARKNET_LIBS /home/ryo/Lib/darknet/libdarknet.so)


##Python 
#set(Python_ADDITIONAL_VERSIONS 3.5)
find_package(PythonLibs REQUIRED)
set(MASKGENERATION_RCF_DIR ${CMAKE_SOURCE_DIR}/photometricSegmentation CACHE STRING "Location of RCF")
set(MASKGENERATION_PYTHON_VE_PATH ${CMAKE_SOURCE_DIR}/photometricSegmentation/python-environment CACHE STRING "Python virtual environment to use")
#configure_file("${CMAKE_SOURCE_DIR}/photometricSegmentation/rcf.py.in" "${CMAKE_CURRENT_BINARY_DIR}/rcf.py")
set (PYTHON_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/photometricSegmentation/python-environment/include/python3.5m)
message(${PYTHON_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})

##Debug mode or Release mode

if(CMAKE_BUILD_TYPE MATCHES Debug)
    message("Debug build.")
else(CMAKE_BUILD_TYPE MATCHES Release)
    message("Release build.")
    add_compile_options(-std=c++11)
    add_compile_options(-O3)
    add_compile_options(-ffast-math)
    add_compile_options(-Wall)
    add_compile_options(-Wno-maybe-uninitialized)
endif(CMAKE_BUILD_TYPE MATCHES Debug)

CUDA_ADD_EXECUTABLE(MaskGeneration Source.cpp Header.h 
${mainEngine}
${DatasetReader}
${DimensionConvertor}
${GeometricSegmentation}
${JointBilateralFilter}
${NormalMapGenerator}
${ObjectDetector}
${RGBDSceneReader}
${NYUv2Reader}
${CoFusionReader}
${photometricSegmentation}
${utils}
${header_mainEngine}
${header_DatasetReader}
${header_DimensionConvertor}
${header_GeometricSegmentation}
${header_JointBilateralFilter}
${header_NormalMapGenerator}
${header_ObjectDetector}
${header_RGBDSceneReader}
${header_NYUv2Reader}
${header_CoFusionReader}
${header_photometricSegmentation}
${header_utils}
)
target_compile_options(MaskGeneration PUBLIC ${compile_flags})
TARGET_LINK_LIBRARIES(MaskGeneration ${CUDART_LIBS} ${OpenCV_LIBS} ${DARKNET_LIBS} ${libraries} ${PYTHON_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

