#include <vector>
#include <string>
#include <iostream>
#include "../include/multiclass.hpp"
#include "random_uniform_generator.cpp"
#include "../include/template_neuralnet.hpp"	

int main() {

    //### the below code is used for multi-class classification
	const unsigned int n_samples_train = 30;
	const unsigned int n_samples_valid = 10;

	intVector data_train;
	intVector data_valid;
	intVector data_train2;
	intVector data_valid2;
	intVector train_labels;
	intVector valid_labels;
	std::generate_n(std::back_inserter(data_train), n_samples_train, RandomNumberBetween(0, 79));
	std::generate_n(std::back_inserter(data_valid), n_samples_valid, RandomNumberBetween(0, 79));
	std::generate_n(std::back_inserter(data_train2), n_samples_train, RandomNumberBetween(0, 79));
	std::generate_n(std::back_inserter(data_valid2), n_samples_valid, RandomNumberBetween(0, 79));
	std::generate_n(std::back_inserter(train_labels), n_samples_train, RandomNumberBetween(0, 2));
	std::generate_n(std::back_inserter(valid_labels), n_samples_valid, RandomNumberBetween(0, 2));
	std::vector<Point4D> data_train_mat;
	std::vector<Point4D> data_test_mat;
	std::cout << "size is " << data_train_mat.size() << std::endl;
	for (unsigned int i = 0; i < n_samples_train; ++i) {
		Point4D randomVecTrain(4);
		if (train_labels[i] == 0) {
			randomVecTrain.w = 0;
			randomVecTrain.x = 1;
			randomVecTrain.y = 2;
			randomVecTrain.z = 3;
		}
		if (train_labels[i] == 1) {
			randomVecTrain.w = 2;
			randomVecTrain.x = 3;
			randomVecTrain.y = 4;
			randomVecTrain.z = 5;
		}
		if (train_labels[i] == 2) {
			randomVecTrain.w = 4;
			randomVecTrain.x = 5;
			randomVecTrain.y = 6;
			randomVecTrain.z = 7;
		}

		data_train_mat.push_back(randomVecTrain);

	}
	for (unsigned int i = 0; i < n_samples_valid; ++i) {
		Point4D randomVecTest(4);
		if (valid_labels[i] == 0) {
			randomVecTest.w = 0;
			randomVecTest.x = 1;
			randomVecTest.y = 2;
			randomVecTest.z = 3;
		}
		if (valid_labels[i] == 1) {
			randomVecTest.w = 2;
			randomVecTest.x = 3;
			randomVecTest.y = 4;
			randomVecTest.z = 5;
		}
		if (valid_labels[i] == 2) {
			randomVecTest.w = 4;
			randomVecTest.x = 5;
			randomVecTest.y = 6;
			randomVecTest.z = 7;
		}

		data_test_mat.push_back(randomVecTest);

	}
	unsigned int n_inputs = 4;
	unsigned int n_hidden_1 = 16;
	unsigned int n_hidden_2 = 8;
	unsigned int n_hidden_3 = 6;
	unsigned int n_outputs = 3;
	unsigned int n_epochs = 40;
	double learning_rate = 1e-4;
	std::string mode = "classification";

	templatenet<std::vector<Point4D>> neural_network(n_inputs, n_hidden_1, n_hidden_2, n_hidden_3, n_outputs, n_epochs, learning_rate, mode);
	neural_network.run(data_train_mat, data_test_mat, train_labels, valid_labels);

	//### the below code is used for binary classification
	/*const unsigned int n_samples_train = 300;
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

	neural_network.run(data_train, data_valid, train_labels, valid_labels);*/

	getchar();
	return 0;
}
