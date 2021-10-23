# MLP_Template
Experiment with creating an MLP from scratch in C++
## WIP
- supports multiple hidden layers
- suports both classification and regression using softmax outputs or sigmoid outputs depending on the task
## Requirements
- C++ 17
- Built using MSVC
## File informaion
- src/neural_network_multilclass.cpp can be used for multi-class predictions
- src/neural_network_regression.cpp can be used for both regression and binary classificaiton
## Future Work
- finish creation of CMake files to allow for compilation on Linux and OSX
- Re-structure proect to have a base class which contains common functionality that can be inherited by both the classification and regression classes
