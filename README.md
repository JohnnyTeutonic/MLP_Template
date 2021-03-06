# MLP_Template
Experiment with creating an MLP from scratch in C++
## Supported Features
- supports multiple hidden layers
- suports multi-class and binary classification
## Requirements
- A compiler with compatibility and support for C++ 17 - either MSVC 2017 (Windows) or GCC >= 5.0 (Linux/OSX)
## File information
- include/multiclass.hpp can be used for multi-class classification and binary classification and is the main class to use for this project.
- include/template_neuralnet.hpp is the same code as above but has been made generic by using templates - this means you can use feature vectors of arbitrary numeric types and it also contains functionality for using drop-out on hidden layers.
## Build instructions for Windows
- from the root directory of the project, using a command prompt equipped with MSVC 2017, run the following commands:
```
mkdir build && cd build	
```
For an x64 build:
```
cmake -G "Visual Studio 15 2017" -A x64 -S ../
cmake --build .
```
Or for an x86 build:
```
cmake -G "Visual Studio 15 2017" -A Win32 -S ../
cmake --build .
```
- this will generate an executable in build/debug/
- the executable is a demonstration of running an MLP multi-class classification problem using synthetic data with 3 hidden layers
- alternatively, you can simply run a script using a command prompt equipped with MSVC 2017 by running the following command from the root directory:
```
build_project_windows.bat
```
- this will also generate an executable in build/debug/

## Build instructions for Linux
- from the root directory of the project run:
```
mkdir build_linux && cd build_linux
cmake .. -G"Unix Makefiles"
make
```
- this will generate an executable in "build_linux" named "updated_mlp"
- alternatively, you can use an automated shell script:
```
./build_project_linus.sh
```
- this will also generate an executable in "build_linux" named "updated_mlp"

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

- 'template_neuralnet.hpp' contains the most-up-to-date code and includes drop-out functionality within. 
- An example of how to perform multi-class classification (using template_neuralnet.hpp) with drop-out on individual hidden layers, using a feature vector of ints, is also found in main.cpp and shown below:
```
	const unsigned int n_samples_train = 100;
	const unsigned int n_samples_valid = 10;

	intVector train_labels;
	intVector valid_labels;
	std::generate_n(std::back_inserter(train_labels), n_samples_train, RandomNumberBetween(0, 2));
	std::generate_n(std::back_inserter(valid_labels), n_samples_valid, RandomNumberBetween(0, 2));
	std::vector<intVector> data_train_mat_vec;
	std::vector<intVector> data_valid_mat_vec;
	for (unsigned int i = 0; i < n_samples_train; ++i) {
		intVector data_train_temp;
		std::generate_n(std::back_inserter(data_train_temp), 6, RandomNumberBetween(0, 29));
		data_train_mat_vec.push_back(data_train_temp);
	}

	for (unsigned int i = 0; i < n_samples_valid; ++i) {
		intVector data_valid_temp;
		std::generate_n(std::back_inserter(data_valid_temp), 6, RandomNumberBetween(0, 29));
		data_valid_mat_vec.push_back(data_valid_temp);

	}

	unsigned int n_inputs = 6;
	unsigned int n_hidden_1 = 16;
	unsigned int n_hidden_2 = 8;
	unsigned int n_hidden_3 = 6;
	unsigned int n_outputs = 3;
	unsigned int n_epochs = 40;
	double learning_rate = 1e-4;
	std::string mode = "classification";
	doubleVector drop_probs = { 0.5, 0.0, 0.0 };
	templatenet<std::vector<intVector>> neural_network(n_inputs, n_hidden_1, n_hidden_2, n_hidden_3, n_outputs, n_epochs, learning_rate, drop_probs, mode);
	neural_network.run(data_train_mat_vec, data_valid_mat_vec, train_labels, valid_labels);
```

## Future Work
- Add in functionality for regression - at the moment only classification (both binary and multi-class) is supported
- Add in functionality to support an arbitrary number of hidden inputs - at the moment the code is hard-coded for 3 different hidden layers.
