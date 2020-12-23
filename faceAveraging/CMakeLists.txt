cmake_minimum_required(VERSION 3.17)
project(faceAveraging)

set(CMAKE_CXX_STANDARD 17)

SET (OpenCV_STATIC ON)

find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(dlib REQUIRED)


add_executable(faceAveraging main.cpp)

TARGET_LINK_LIBRARIES(faceAveraging ${OpenCV_LIBS} dlib::dlib )