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
	struct Point {
		double x;
		double y;
		unsigned int sz;
		Point::Point(unsigned int size) : sz(size) {};
	};
	std::vector<Point> data_train_mat;
	std::vector<Point> data_test_mat;
	std::cout << "size is " << data_train_mat.size() << std::endl;
	for (unsigned int i = 0; i < 30; ++i) {
		Point randomVecTrain(2);
		if (train_labels[i] == 0) {
			randomVecTrain.x = 0;
			randomVecTrain.y = 1;
		}
		if (train_labels[i] == 1) {
			randomVecTrain.x = 3;
			randomVecTrain.y = 4;
		}
		if (train_labels[i] == 2) {
			randomVecTrain.x = 5;
			randomVecTrain.y = 6;
		}

		data_train_mat.push_back(randomVecTrain);

	}
	for (unsigned int i = 0; i < 10; ++i) {
		Point randomVecTest(2);
		if (valid_labels[i] == 0) {
			randomVecTest.x = 0;
			randomVecTest.y = 1;
		}
		if (valid_labels[i] == 1) {
			randomVecTest.x = 3;
			randomVecTest.y = 4;
		}
		if (valid_labels[i] == 2) {
			randomVecTest.x = 5;
			randomVecTest.y = 6;
		}

		data_test_mat.push_back(randomVecTest);

	}
	unsigned int n_inputs = 2;
	unsigned int n_hidden_1 = 16;
	unsigned int n_hidden_2 = 8;
	unsigned int n_hidden_3 = 6;
	unsigned int n_outputs = 3;
	unsigned int n_epochs = 40;
	double learning_rate = 1e-4;

	templatenet<std::vector<Point>> neural_network(n_inputs, n_hidden_1, n_hidden_2, n_hidden_3, n_outputs, n_epochs, learning_rate);


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
