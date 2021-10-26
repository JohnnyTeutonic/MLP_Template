# MLP_Template
Experiment with creating an MLP from scratch in C++
## Supported Features
- supports multiple hidden layers
- suports multi-class and binary classification
## Requirements
- C++ 17
- Built using MSVC
## File information
- include/multiclass.hpp can be used for multi-class classification and binary classification and is the main class to use for this project.
- include/template_neuralnet.hpp is the same code as above but has been made generic by using templates - this means you can use feature vectors of arbitrary numeric types.
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
## Example for multi-class classification (using synthetic data)
- there is example code in 'main.cpp' of how to use 'multiclass.hpp' or 'template_neuralnet.hpp' for both binary and multi-class classification.
- for multi-class classification, you can use something like the below code (please note that this is using random data):
```
	const unsigned int n_samples_train = 300;
	const unsigned int n_samples_valid = 50;

	intVector data_train;
	intVector data_valid;
	intVector train_labels;
	intVector valid_labels;
	std::generate_n(std::back_inserter(data_train), n_samples_train, RandomNumberBetween(0, 49));
	std::generate_n(std::back_inserter(data_valid), n_samples_valid, RandomNumberBetween(0, 49));
	std::generate_n(std::back_inserter(train_labels), n_samples_train, RandomNumberBetween(0, 2));
	std::generate_n(std::back_inserter(valid_labels), n_samples_valid, RandomNumberBetween(0, 2));
	unsigned int n_inputs = 50;
	unsigned int n_hidden_1 = 16;
	unsigned int n_hidden_2 = 8;
	unsigned int n_hidden_3 = 6;
	unsigned int n_outputs = 3;
	unsigned int n_epochs = 60;
	double learning_rate = 1e-4;

	experimental neural_network(n_inputs, n_hidden_1, n_hidden_2, n_hidden_3, n_outputs, n_epochs, learning_rate);
```
- for binary classification, you can similarly construct an example like this:
```
        const unsigned int n_samples_train = 300;
	const unsigned int n_samples_valid = 50;

	intVector data_train;
	intVector data_valid;
	intVector train_labels;
	intVector valid_labels;
	std::generate_n(std::back_inserter(data_train), n_samples_train, RandomNumberBetween(0, 49));
	std::generate_n(std::back_inserter(data_valid), n_samples_valid, RandomNumberBetween(0, 49));
	std::generate_n(std::back_inserter(train_labels), n_samples_train, RandomNumberBetween(0, 1));
	std::generate_n(std::back_inserter(valid_labels), n_samples_valid, RandomNumberBetween(0, 1));
	unsigned int n_inputs = 50;
	unsigned int n_hidden_1 = 16;
	unsigned int n_hidden_2 = 8;
	unsigned int n_hidden_3 = 6;
	unsigned int n_outputs = 1;
	unsigned int n_epochs = 60;
	double learning_rate = 1e-4;

	experimental neural_network(n_inputs, n_hidden_1, n_hidden_2, n_hidden_3, n_outputs, n_epochs, learning_rate);

	neural_network.run(data_train, data_valid, train_labels, valid_labels);
```

## Future Work
- Add in functionality for regression - at the moment only classification (both binary and multi-class) is supported
- Add in functionality to support arbitrary number of hidden inputs - at the moment the code is hard-coded for 3 different hidden layers.
- Add in functionality to support cross-compilation - only Windows is supported at the moment
