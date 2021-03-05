cmake_minimum_required(VERSION 3.17)
project(tesseract_ocr_sample)

set(CMAKE_CXX_STANDARD 17)

file(COPY tesseract-snapshot.png DESTINATION ${CMAKE_BINARY_DIR})

find_package(Tesseract CONFIG REQUIRED)
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(tesseract_ocr_sample  PRIVATE libtesseract  ${OpenCV_LIBS})


