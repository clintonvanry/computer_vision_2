cmake_minimum_required(VERSION 3.17)
project(imgClassCNNWithLibTorch)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_PREFIX_PATH C://libtorch/share/cmake/Torch)

find_package(Torch REQUIRED)
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(imgClassCNNWithLibTorch main.cpp)

target_link_libraries(imgClassCNNWithLibTorch ${OpenCV_LIBS})
target_link_libraries(imgClassCNNWithLibTorch "${TORCH_LIBRARIES}")

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET imgClassCNNWithLibTorch
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:imgClassCNNWithLibTorch>)
endif (MSVC)