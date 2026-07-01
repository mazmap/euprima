cmake_minimum_required(VERSION 3.12)
project(euprima)

# To suppress the warning for using pack fold expressions (C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the pybind11 package
find_package(pybind11 REQUIRED)

# Create the extension module
pybind11_add_module(euprima cppbinding/profiles.cpp)

target_compile_options(euprima PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-O3 -march=native>
    $<$<CXX_COMPILER_ID:Clang>:-O3 -march=native>
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /arch:AVX2>
)
