cmake_minimum_required(VERSION 3.17)
project(assignment3_ocr_on_invoice)

set(CMAKE_CXX_STANDARD 17)

file(COPY invoice.jpg  DESTINATION ${CMAKE_BINARY_DIR})

find_package(Tesseract CONFIG REQUIRED)
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME}  PRIVATE libtesseract  ${OpenCV_LIBS})