cmake_minimum_required (VERSION 3.8)

project (updated_mlp VERSION 0.5 DESCRIPTION "A minimal example of a vanilla MLP for Regression and Classification" LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
include_directories("${PROJECT_SOURCE_DIR}")
include_directories(
		${PROJECT_SOURCE_DIR}/include
		${PROJECT_SOURCE_DIR}/src
)

file(GLOB all_SRCS
        "${PROJECT_SOURCE_DIR}/include/*.h"
		"${PROJECT_SOURCE_DIR}/include/*.hpp"
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
		"${PROJECT_SOURCE_DIR}/src/*.c"
        )

set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard is 17")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_executable(updated_mlp ${all_SRCS})

target_compile_features(updated_mlp PUBLIC cxx_std_17)
set_target_properties(updated_mlp PROPERTIES CXX_EXTENSIONS OFF)
