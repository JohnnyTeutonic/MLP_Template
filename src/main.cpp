#include <vector>
#include <string>
#include <iostream>
#include "../include/multiclass.hpp"
#include "random_uniform_generator.cpp"


int main() {

	const unsigned int n_samples_train = 300;
	const unsigned int n_samples_valid = 50;
	const unsigned int n_class_labels = 3;
	intVector data_train;
	intVector data_valid;
	intVector class_samples;
	std::generate_n(std::back_inserter(data_train), n_samples_train, RandomNumberBetween(0, 49));
	std::generate_n(std::back_inserter(data_valid), n_samples_valid, RandomNumberBetween(0, 2));
	std::generate_n(std::back_inserter(class_samples), n_class_labels, RandomNumberBetween(0, 2));
	unsigned int n_inputs = 50;
	unsigned int n_hidden_1 = 16;
	unsigned int n_hidden_2 = 8;
	unsigned int n_hidden_3 = 6;
	unsigned int n_outputs = 3;
	unsigned int n_epochs = 60;
	double learning_rate = 1e-4;

	experimental neural_network(n_inputs, n_hidden_1, n_hidden_2, n_hidden_3, n_outputs, n_epochs, learning_rate);

	neural_network.run(data_train, data_valid, class_samples);
	getchar();
	return 0;
}
