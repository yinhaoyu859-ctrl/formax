cmake_minimum_required(VERSION 3.10)
project(Test)

find_package(OpenCV REQUIRED)

add_subdirectory(common)
add_subdirectory(modules)

add_executable(test main.cpp)
target_link_libraries(test M1 M2 math)