cmake_minimum_required(VERSION 3.15)
project(euprima LANGUAGES CXX)

find_package(pybind11 REQUIRED)

pybind11_add_module(_core cppbinding/profiles.cpp)

target_compile_options(_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-O3>
    $<$<CXX_COMPILER_ID:Clang>:-O3>
    $<$<CXX_COMPILER_ID:MSVC>:/O2>
)

install(TARGETS _core DESTINATION euprima)
