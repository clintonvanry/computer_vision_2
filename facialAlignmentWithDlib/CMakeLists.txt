cmake_minimum_required(VERSION 3.17)
project(facialAlignmentWithDlib)

set(CMAKE_CXX_STANDARD 17)

SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
SET (OpenCV_STATIC ON)

find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(dlib REQUIRED)

add_executable(facialAlignmentWithDlib main.cpp faceBlendCommon.hpp)

TARGET_LINK_LIBRARIES(facialAlignmentWithDlib ${OpenCV_LIBS} dlib::dlib )