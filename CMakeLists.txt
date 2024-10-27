cmake_minimum_required(VERSION 3.10)
project(opencl_vector_add)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OCL_SDK_PATH "C:/Program Files (x86)/OCL_SDK_Light")

include_directories(${OCL_SDK_PATH}/include)

link_directories(${OCL_SDK_PATH}/lib/x86_64)

add_executable(${PROJECT_NAME} 
    frist_test/main.cpp
)

target_link_libraries(${PROJECT_NAME} OpenCL)

configure_file(${CMAKE_SOURCE_DIR}/frist_test/vecAdd.cl
               ${CMAKE_BINARY_DIR}/vecAdd.cl COPYONLY)
