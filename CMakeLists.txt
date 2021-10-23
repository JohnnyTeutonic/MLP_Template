
cmake_minimum_required (VERSION 3.8)

project (updated_mlp VERSION 0.5 DESCRIPTION "A minimal example of a vanilla MLP for Regression and Classification" LANGUAGES cxx_std_17)
add_subdirectory(src)
add_subdirectory(include)
#add_executable(main updated_mlp/src/main.cpp)
add_executable(main "$projectname$/src/main.cpp)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard is 17")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# target_compile_features(updated_mlp PUBLIC cxx_std_17)
# set_target_properties(updated_mlp PROPERTIES CXX_EXTENSIONS OFF)
