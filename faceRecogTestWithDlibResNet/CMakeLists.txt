cmake_minimum_required(VERSION 3.17)
project(faceRecogTestWithDlibResNet)

SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
SET (OpenCV_STATIC ON)

find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(dlib REQUIRED)

add_executable(faceRecogTestWithDlibResNet main.cpp)

TARGET_LINK_LIBRARIES(faceRecogTestWithDlibResNet ${OpenCV_LIBS} dlib::dlib )