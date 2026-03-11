cmake_minimum_required(VERSION 3.12)
project(euprima)

# Find the pybind11 package
find_package(pybind11 REQUIRED)

# Create the extension module
pybind11_add_module(euprima cppbinding/profiles.cpp)
