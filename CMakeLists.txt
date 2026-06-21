cmake_minimum_required(VERSION 3.26)
project(DeepBit)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)
include_directories(include/hnswlib)
# Eigen3 is a header-only library; Ubuntu installs it under eigen3/ prefix
find_package(Eigen3 QUIET)
if(EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
else()
    include_directories(/usr/include/eigen3)
endif()
find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()

add_definitions(-std=c++17 -Ofast -march=core-avx2 -mavx512f -fpic -fopenmp -ftree-vectorize -fexceptions)
add_subdirectory(src)