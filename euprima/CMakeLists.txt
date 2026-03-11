cmake_minimum_required(VERSION 3.12)
project(euprima)

# To suppress the warning for using pack fold expressions (C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the pybind11 package
find_package(pybind11 REQUIRED)

# Create the extension module
pybind11_add_module(euprima cppbinding/profiles.cpp)
