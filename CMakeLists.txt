cmake_minimum_required(VERSION 3.14)
project(euprima LANGUAGES CXX)

find_package(pybind11 REQUIRED)

pybind11_add_module(euprima cppbinding/profiles.cpp)

target_compile_options(euprima PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-O3>
    $<$<CXX_COMPILER_ID:Clang>:-O3>
    $<$<CXX_COMPILER_ID:MSVC>:/O2>
)
