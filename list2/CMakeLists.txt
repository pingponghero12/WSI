cmake_minimum_required(VERSION 3.14)
project(puzzle_solver CXX)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENTIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -O3")

include_directories(include)

file(GLOB SOURCE "src/*.cpp")
add_executable(solver ${SOURCE})
