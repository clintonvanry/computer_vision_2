cmake_minimum_required(VERSION 3.17)
project(trainingFacialLandmarkDectorWithDlib)

set(CMAKE_CXX_STANDARD 17)

SET (OpenCV_STATIC ON)

find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(dlib REQUIRED)

add_executable(trainingFacialLandmarkDectorWithDlib main.cpp)

TARGET_LINK_LIBRARIES(trainingFacialLandmarkDectorWithDlib ${OpenCV_LIBS} dlib::dlib )