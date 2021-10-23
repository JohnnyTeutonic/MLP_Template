
cmake_minimum_required (VERSION 3.8)

project ("updated_mlp")

add_subdirectory ("$projectname$/src")
add_subdirectory ("$projectname$/include")

add_executable("$projectname$/src/main.cpp)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard is 17")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# target_compile_features(updated_mlp PUBLIC cxx_std_17)
# set_target_properties(updated_mlp PROPERTIES CXX_EXTENSIONS OFF)
