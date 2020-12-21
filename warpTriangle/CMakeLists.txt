cmake_minimum_required(VERSION 3.17)
project(warpTriangle)

set(CMAKE_CXX_STANDARD 17)

SET (OpenCV_STATIC ON)

find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(warpTriangle main.cpp)

TARGET_LINK_LIBRARIES(warpTriangle ${OpenCV_LIBS} )