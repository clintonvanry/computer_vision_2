cmake_minimum_required(VERSION 3.17)
project(project1_virtualmakeup_lips)

set(CMAKE_CXX_STANDARD 17)

SET (OpenCV_STATIC ON)

find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(dlib REQUIRED)

add_executable(project1_virtualmakeup_lips main.cpp faceBlendCommon.hpp)

TARGET_LINK_LIBRARIES(project1_virtualmakeup_lips ${OpenCV_LIBS} dlib::dlib )