#include <cmath>
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <stdlib.h>	
#include <vector>
#include "../include/utils.hpp"
#include "../include/random_index_generator.hpp"
#include "../include/multiclass.hpp"


experimental::experimental(unsigned int _n_inputs,
	unsigned int _n_hidden_1,
	unsigned int _n_hidden_2,
	unsigned int _n_hidden_3,
	unsigned int _n_outputs,
	unsigned int _n_epochs,
	double _learning_rate) : gen{ std::random_device()() } {

	n_inputs = _n_inputs;
	n_hidden_1 = _n_hidden_1;
	n_hidden_2 = _n_hidden_2;
	n_hidden_3 = _n_hidden_3;
	n_outputs = _n_outputs;
	n_epochs = _n_epochs;
	learning_rate = _learning_rate;

	W1.resize(n_hidden_1, doubleVector(n_inputs, 0.0));
	W2.resize(n_hidden_2, doubleVector(n_hidden_1, 0.0));
	W3.resize(n_hidden_3, doubleVector(n_hidden_2, 0.0));
	W4.resize(n_outputs, doubleVector(n_hidden_3, 0.0));
	he_initialization(W1);
	he_initialization(W2);
	he_initialization(W3);
	he_initialization(W4);

	b1.resize(n_hidden_1, 0.0);
	b2.resize(n_hidden_2, 0.0);
	b3.resize(n_hidden_3, 0.0);
	b4.resize(n_outputs, 0.0);

	z1.resize(n_hidden_1);
	z2.resize(n_hidden_2);
	z3.resize(n_hidden_3);
	z4.resize(n_outputs);

	x.resize(n_inputs);
	x1.resize(n_hidden_1);
	x2.resize(n_hidden_2);
	x3.resize(n_hidden_3);
	x4.resize(n_outputs);
	y.resize(n_outputs);

	delta1.resize(n_hidden_1);
	delta2.resize(n_hidden_2);
	delta3.resize(n_hidden_3);
	delta4.resize(n_outputs);

	dW1.resize(n_hidden_1, doubleVector(n_inputs, 0.0));
	dW2.resize(n_hidden_2, doubleVector(n_hidden_1, 0.0));
	dW3.resize(n_hidden_3, doubleVector(n_hidden_2, 0.0));
	dW4.resize(n_outputs, doubleVector(n_hidden_3, 0.0));

	db1.resize(n_hidden_1, 0.0);
	db2.resize(n_hidden_2, 0.0);
	db3.resize(n_hidden_3, 0.0);
	db4.resize(n_outputs, 0.0);
}

void experimental::he_initialization(doubleMatrix& W) {
	double mean = 0.0;
	double std = std::sqrt(2.0 / static_cast<double>(W.size()));
	std::normal_distribution<double> rand_normal(mean, std);
	for (unsigned int i = 0; i < W.size(); ++i) {
		for (unsigned int j = 0; j < W[0].size(); ++j) {
			W[i][j] = rand_normal(gen);
		}
	}
}

void experimental::run(intVector& data_train, intVector& data_valid, intVector& class_labels, intVector& valid_labels) {
	auto it = unique(data_train.begin(), data_train.end());
	intVector data_train_2;
	std::copy(data_train.begin(), data_train.end(), std::back_inserter(data_train_2));
	data_train_2.resize(std::distance(data_train.begin(), it));
	auto it2 = unique(class_labels.begin(), class_labels.end());
	intVector class_labels_2;
	std::copy(class_labels.begin(), class_labels.end(), std::back_inserter(class_labels_2));
	class_labels_2.resize(std::distance(class_labels.begin(), it2));

	RandomIndex rand_idx(data_train.size());
	RandomIndex rand_idx2(y.size());
	RandomIndex rand_idx3(x.size());
	unsigned int idx, idx2;
	for (unsigned int i = 0; i < n_epochs; ++i) {
		std::cout << "epoch no. " << i << '\n';
		std::string s(50, '*');
		std::cout << s << std::endl;
		for (unsigned int j = 0; j < data_train.size(); ++j) {
			idx = rand_idx.get();
			auto idx2 = rand_idx2.get();
			auto idx3 = rand_idx3.get();
			x[idx3] = double(data_train[idx]);
			std::fill(y.begin(), y.end(), 0.0);
			y[idx2] = class_labels[idx];
			feedforward();
			backpropagation();
			gradient_descent();
		}

		if (i % 2 == 0) {
			comp_stats(data_valid, valid_labels);
		}
	}
}

void experimental::feedforward() {
	z1 = matmul(W1, x, b1);
	x1 = relu(z1);
	z2 = matmul(W2, x1, b2);
	x2 = relu(z2);
	z3 = matmul(W3, x2, b3);
	x3 = relu(z3);
	z4 = matmul(W4, x3, b4);
	x4 = softmaxoverflow(z4);
}

void experimental::comp_gradients(doubleMatrix& dW,
	doubleVector& db,
	const doubleVector& x,
	const doubleVector& delta) {
	for (unsigned int i = 0; i < dW.size(); ++i) {
		for (unsigned int j = 0; j < dW[0].size(); ++j) {
			dW[i][j] = x[j] * delta[i];
		}
		db[i] = delta[i];
	}
}

void experimental::comp_delta_init(doubleVector& delta,
	const doubleVector& z,
	const doubleVector& x,
	const doubleVector& y) {
	for (unsigned int i = 0; i < delta.size(); ++i) {
		delta[i] = softmax_prime_single(i);
	}
}

void experimental::comp_delta(const doubleMatrix& W,
	const doubleVector& z,
	const doubleVector& delta_old,
	doubleVector& delta) {
	for (unsigned int j = 0; j < W[0].size(); ++j) {
		double tmp = 0.0;
		for (unsigned int i = 0; i < W.size(); ++i) {
			tmp += W[i][j] * delta_old[i];
		}
		delta[j] = relu_prime(z[j]) * tmp;
	}
}


void experimental::backpropagation() {
	comp_delta_init(delta4, z4, x4, y);
	comp_gradients(dW4, db4, x3, delta4);

	comp_delta(W4, z3, delta4, delta3);
	comp_gradients(dW3, db3, x2, delta3);

	comp_delta(W3, z2, delta3, delta2);
	comp_gradients(dW2, db2, x1, delta2);

	comp_delta(W2, z1, delta2, delta1);
	comp_gradients(dW1, db1, x, delta1);
}

void experimental::gradient_descent() {
	descent(W4, b4, dW4, db4);
	descent(W3, b3, dW3, db3);
	descent(W2, b2, dW2, db2);
	descent(W1, b1, dW1, db1);
}

void experimental::descent(doubleMatrix& W,
	doubleVector& b,
	const doubleMatrix& dW,
	const doubleVector& db) {
	for (unsigned int i = 0; i < W.size(); ++i) {
		for (unsigned int j = 0; j < W[0].size(); ++j) {
			W[i][j] -= learning_rate * dW[i][j];
		}
		b[i] -= learning_rate * db[i];
	}
}

double experimental::loss_function_cross_entropy(double epsilon = 1e-8) {
	doubleVector loss_vec;
	transform(y.begin(), y.end(), x4.begin(), std::back_inserter(loss_vec), [&](double x, double y) {return x * std::log(y + epsilon); });
	double loss = std::accumulate(loss_vec.begin(), loss_vec.end(), 0.0);
	return -loss;
}


double experimental::comp_accuracy() {
	double accuracy = 0.0;
	unsigned int prediction = std::distance(x4.begin(), std::max_element(x4.begin(), x4.end()));
	unsigned int ground_truth = std::distance(y.begin(), std::max_element(y.begin(), y.end()));
	if (prediction == ground_truth) {
		accuracy += 1.0;
	}
	return accuracy;
}

void experimental::comp_stats(const intVector& data, const intVector& labels) {
	double loss = 0.0;
	double accuracy = 0.0;
	for (unsigned int i = 0; i < data.size(); ++i) {
		std::fill(y.begin(), y.end(), 0);
		x[i] = double(data[i]);
		y[labels[i]] = labels[i];
		feedforward();
		loss += loss_function_cross_entropy();
		accuracy += comp_accuracy();
	}
	loss /= static_cast<double>(data.size());
	accuracy /= static_cast<double>(data.size());
	std::cout << "loss is: " << loss << " accuracy is: " << accuracy << std::endl;
}

doubleVector experimental::comp_prediction(const intVector& preds) {
	doubleVector prediction(preds.size());
	for (unsigned int i = 0; i < preds.size(); ++i) {
		x[i] = double(preds[i]);
		feedforward();
		prediction[i] = x4[0];
	}
	return prediction;
}


doubleVector experimental::convert_probs_to_class(doubleVector & probs) {
	doubleVector::iterator result = max_element(probs.begin(), probs.end());
	int argmaxVal = distance(probs.begin(), result);
	int selected_class = static_cast<int>(probs[argmaxVal]);
	doubleVector one_hot_classes(probs.size());
	fill(one_hot_classes.begin(), one_hot_classes.end(), 0.0);
	one_hot_classes[selected_class] = 1.0;
	return one_hot_classes;
}


doubleVector experimental::softmaxoverflow(doubleVector & weights) {
	doubleVector secondweights, sum;
	double max = *max_element(weights.begin(), weights.end()); // use the max value to handle overflow issues

	for (unsigned int i = 0; i < weights.size(); i++) {
		sum.emplace_back(exp(weights[i] - max));
	}

	double norm2 = std::accumulate(sum.begin(), sum.end(), 0.0);

	for (unsigned int i = 0; i < weights.size(); i++) {
		secondweights.emplace_back(exp(weights[i] - max) / norm2);
	}
	return secondweights;
}



doubleVector experimental::matmul(const doubleMatrix& W, const doubleVector& x, const doubleVector& b) 
{
	doubleVector z(W.size(), 0.0);

	for (unsigned int i = 0; i < W.size(); ++i) {

		for (unsigned int j = 0; j < W[0].size(); ++j) {
			z[i] += W[i][j] * x[j];
		}
		z[i] += b[i];
	}
	return z;
}

doubleVector experimental::relu(const doubleVector& z) {
	doubleVector x(z.size());
	for (unsigned int i = 0; i < z.size(); ++i) {
		x[i] = (z[i] >= 0.0 ? z[i] : 0.0);
	};
	return x;
}

double experimental::relu_prime(const double z) {
	return (z >= 0.0 ? 1.0 : 0.0);
}


double experimental::sigmoid_prime(const double z, bool first = true) {
	const double denominator = (1.0 + std::exp(-z));
	const double denominator2 = (1.0 + std::exp(z));
	double sigma;
	sigma = (z > 0.0) ? (1.0 / denominator) : (std::exp(z) / denominator2);
	if (first == true) {
		{ return sigma * (1.0 - sigma); };

	};
	return sigma;
}


doubleVector experimental::sigmoid(const doubleVector& z) {
	doubleVector x(z.size());
	for (unsigned int i = 0; i < z.size(); ++i) {
		x[i] = sigmoid_prime(z[i], false);
	}
	return x;
}

inline double experimental::softmax_prime_single(unsigned int index) {
	double result;
	result = x4[index] - y[index];
	return result;
}


double experimental::softmax_prime() {
	double result;
	for (unsigned int i = 0; i < y.size(); ++i) {
		result += x4[i] - y[i];
	}
	return result;
};
