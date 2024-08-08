cmake_minimum_required(VERSION 3.0)

set(CMAKE_BUILD_TYPE Debug)

project(MyProject)

# Include header files directory
include_directories(${CMAKE_SOURCE_DIR}/headers)

file (GLOB SOURCES "headers/*.cpp" "headers/src/*.cpp")

add_executable(runner.out ${SOURCES} main.cpp)
