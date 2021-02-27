cmake_minimum_required(VERSION 3.17)
project(linear_regressionWithLibTorch)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_PREFIX_PATH C://libtorch/share/cmake/Torch)

file(COPY BostonHousing.csv DESTINATION ${CMAKE_BINARY_DIR})

find_package(Torch REQUIRED)

add_executable(linear_regressionWithLibTorch main.cpp)
target_include_directories(linear_regressionWithLibTorch PRIVATE "C://boost/boost_1_75_0")
target_link_libraries(linear_regressionWithLibTorch "${TORCH_LIBRARIES}")

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET linear_regressionWithLibTorch
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:linear_regressionWithLibTorch>)
endif (MSVC)