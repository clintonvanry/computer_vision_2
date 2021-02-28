cmake_minimum_required(VERSION 3.17)
project(assignment2_cifar10_improved_cnn)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_PREFIX_PATH C://libtorch/share/cmake/Torch)

find_package(Torch REQUIRED)

add_executable(assignment2_cifar10_improved_cnn main.cpp)
target_link_libraries(assignment2_cifar10_improved_cnn "${TORCH_LIBRARIES}")

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET assignment2_cifar10_improved_cnn
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:assignment2_cifar10_improved_cnn>)
endif (MSVC)