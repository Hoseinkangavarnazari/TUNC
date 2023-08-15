cmake_minimum_required(VERSION 3.0)

set(CMAKE_BUILD_TYPE Debug)

project(MyProject)

file (GLOB SOURCES "headers/src/*.cpp")

add_executable(runner.out ${SOURCES} main.cpp)
