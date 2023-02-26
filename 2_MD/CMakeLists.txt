cmake_minimum_required(VERSION 3.14.5)

project(my_project_name CXX)

add_executable(a.out main.cpp)

target_compile_options(a.out PRIVATE -O3 -xCORE_AVX512)
target_compile_features(a.out PRIVATE cxx_std_17)
