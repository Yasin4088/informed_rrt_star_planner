cmake_minimum_required(VERSION 3.10)
project(informed_rrt_star_full_pipeline_benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Benchmark executable
add_executable(full_pipeline_benchmark src/main.cpp)
target_include_directories(full_pipeline_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(full_pipeline_benchmark Eigen3::Eigen)

# Aggressive optimizations for fair benchmark comparison
target_compile_options(full_pipeline_benchmark PRIVATE -O3 -march=native -ffast-math)
