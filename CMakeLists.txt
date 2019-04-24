cmake_minimum_required(VERSION 3.1)
project(vector_add)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Handle OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})
add_executable(convolution main.cpp convolution.cl)
add_executable(gen_test gen_test.cpp)
target_link_libraries(convolution ${OpenCL_LIBRARY})

# copy cl files to binary folder
FILE(GLOB CL_FILES *.cl)
foreach (cl_file ${CL_FILES})
    add_custom_command(
            TARGET convolution
            POST_BUILD
            COMMAND ${CMAKE_COMMAND}
            ARGS -E copy ${cl_file} ${PROJECT_BINARY_DIR}/.
    )
endforeach (cl_file)