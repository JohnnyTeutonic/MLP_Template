# MLP_Template
Experiment with creating an MLP from scratch in C++
## WIP
- supports multiple hidden layers
- suports both classification and regression using softmax outputs or sigmoid outputs depending on the task
## Requirements
- C++ 17
- Built using MSVC
## File information
- src/neural_network_multilclass.cpp can be used for multi-class classification
- src/neural_network_regression.cpp can be used for both regression and binary classificaiton
## Build instructions on Windows
- from the root directory of the project, using the Visual Studio Developer command prompt, run the following commands:
```
mkdir build && cd build
cmake -G "Visual Studio 15 2017" ..
cmake --build .
```
- this will generate an executable in build/debug/
- the executable is a demonstration of running an MLP multi-class classification problem using synthetic data with 3 hidden layers
- alternatively, using the Visual Studio Developer command prompt, run the following command from the root dir:
```
sh build_project.sh
```
- this will also generate an executable in build/debug/
## Future Work
- Re-structure project to have a base MLP class which contains common functionality that can be inherited by both the classification and regression MLP classes
