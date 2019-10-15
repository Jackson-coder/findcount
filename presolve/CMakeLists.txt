cmake_minimum_required(VERSION 3.0.0)
project(presolve VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(OpenCV 4.0.0 REQUIRED)


add_executable(presolve main.cpp)
target_link_libraries(presolve ${OpenCV_LIBS})