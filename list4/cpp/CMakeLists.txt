cmake_minimum_required(VERSION 3.10)
project(cl CXX)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(POLICY CMP0148)
    cmake_policy(SET CMP0148 NEW)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(pybind11 REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
set(PYBIND11_FINDPYTHON ON)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -flto=auto -O3")
set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -O3")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

pybind11_add_module(cl_module 
    src/kmeans.cpp 
    src/dbscan.cpp
    src/cl_module.cpp
)
