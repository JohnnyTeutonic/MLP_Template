#pragma once
#include <cassert>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <stdlib.h>	
#include <string>
#include <utility>
#include <vector>
#include <ostream>
#include <string>
#include <random>
#include <cstdlib>
#include <ctime>
#include <type_traits>

#include "utils.hpp"
#include "random_index_generator.hpp"

template<class T>
class templatenet {
public:
	templatenet<T>(unsigned int _n_inputs,
		unsigned int _n_hidden_1,
		unsigned int _n_hidden_2,
		unsigned int _n_hidden_3,
		unsigned int _n_outputs,
		unsigned int _n_epochs,
		double _learning_rate, doubleVector _dropout_probs, std::string mode);

	using actualType = T;
	void run(actualType& data_train, actualType& data_valid, intVector& train_labels, intVector& valid_labels);
	std::mt19937 gen;
	unsigned int n_epochs;
	unsigned int n_inputs;
	unsigned int n_hidden_1;
	unsigned int n_hidden_2;
	unsigned int n_hidden_3;
	unsigned int n_outputs;
	double learning_rate;
	doubleVector dropout_probs;
	std::string mode;

private:
	doubleMatrix W1, W2, W3, W4;
	doubleVector b1, b2, b3, b4;

	doubleMatrix dW1, dW2, dW3, dW4;
	doubleVector db1, db2, db3, db4;

	doubleVector z1, z2, z3, z4;

	doubleVector x, x1, x2, x3, x4, y;

	doubleVector delta1, delta2, delta3, delta4;

	void he_initialization(doubleMatrix& weights);
	void feedforward();

	doubleVector matmul(const doubleMatrix& W,
		const doubleVector& x,
		const doubleVector& b);

	doubleVector relu(const doubleVector& x);
	doubleVector sigmoid(const doubleVector& x);
	double softmax_prime();
	inline double softmax_prime_single(unsigned int index);

	double relu_prime(const double z);
	double sigmoid_prime(const double z, bool first);

	void backpropagation();

	void comp_delta_init(doubleVector& delta,
		const doubleVector& z,
		const doubleVector& x,
		const doubleVector& y);

	void comp_delta(const doubleMatrix& W,
		const doubleVector& z,
		const doubleVector& delta_old,
		doubleVector& delta);

	void comp_gradients(doubleMatrix& dW,
		doubleVector& db,
		const doubleVector& x,
		const doubleVector& delta);

	void gradient_descent();

	void descent(doubleMatrix& W,
		doubleVector& b,
		const doubleMatrix& dW,
		const doubleVector& db);

	void comp_stats(const actualType& data, const intVector& labels);
	double comp_accuracy();
	doubleVector comp_prediction(const actualType& preds);
	doubleVector softmaxoverflow(doubleVector & weights);
	doubleVector convert_probs_to_class(doubleVector & probs);
	double loss_function_cross_entropy(double epsilon);
	double comp_loss_mse();
	void drop_out(double& prob, doubleVector& hidden_layer);
};



template<class T>
templatenet<T>::templatenet(unsigned int _n_inputs,
	unsigned int _n_hidden_1,
	unsigned int _n_hidden_2,
	unsigned int _n_hidden_3,
	unsigned int _n_outputs,
	unsigned int _n_epochs,
	double _learning_rate, doubleVector _dropout_probs, std::string _mode) : gen{ std::random_device()() } {

	n_inputs = _n_inputs;
	n_hidden_1 = _n_hidden_1;
	n_hidden_2 = _n_hidden_2;
	n_hidden_3 = _n_hidden_3;
	n_outputs = _n_outputs;
	n_epochs = _n_epochs;
	learning_rate = _learning_rate;
	dropout_probs = _dropout_probs;
	assert(dropout_probs.size() == 3);
	mode = _mode;
	typedef T value_type;
	value_type actual_type;
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

template<class T>
void templatenet<T>::he_initialization(doubleMatrix& W) {
	double mean = 0.0;
	double std = std::sqrt(2.0 / static_cast<double>(W.size()));
	std::normal_distribution<double> rand_normal(mean, std);
	for (unsigned int i = 0; i < W.size(); ++i) {
		for (unsigned int j = 0; j < W[0].size(); ++j) {
			W[i][j] = rand_normal(gen);
		}
	}
}

template<class T>
void templatenet<T>::drop_out(double& prob, doubleVector& hidden_layer) {
		for (unsigned int i = 0; i < hidden_layer.size(); ++i) {
			float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
			if (r <= prob) {
				hidden_layer[i] = 0;
			}
		}
	}

template<class T>
void templatenet<T>::run(actualType& data_train, actualType& data_valid, intVector& class_labels, intVector& valid_labels) {
	assert(data_train[0].size() == n_inputs);
	RandomIndex rand_idx(data_train.size());
	unsigned int idx;
	using elemType = typename std::decay<decltype(*data_train.begin())>::type;
	for (unsigned int i = 0; i < n_epochs; ++i) {
		std::cout << "epoch no. " << i << '\n';
		std::string s(50, '*');
		std::cout << s << std::endl;
		for (unsigned int j = 0; j < data_train.size(); ++j) {
			idx = rand_idx.get();
			for (unsigned int k = 0; k < data_train[0].size(); ++k) {
				x[k] = static_cast<double>(data_train[idx][k]);
			}
		
			std::fill(y.begin(), y.end(), 0.0);
			if (n_outputs == 1) {
				y[0] = static_cast<double>(class_labels[idx]);
			}
			else {
				y[class_labels[idx]] = 1.0;
			}
			feedforward();
			backpropagation();
			gradient_descent();
		}

		if (i % 2 == 0) {
			comp_stats(data_valid, valid_labels);
		}
	}
}


template<class T>
void templatenet<T>::feedforward() {
	z1 = matmul(W1, x, b1);
	drop_out(dropout_probs[0], z1);
	x1 = relu(z1);
	z2 = matmul(W2, x1, b2);
	drop_out(dropout_probs[1], z2);
	x2 = relu(z2);
	z3 = matmul(W3, x2, b3);
	drop_out(dropout_probs[2], z3);
	x3 = relu(z3);
	z4 = matmul(W4, x3, b4);
	if (n_outputs == 1) { { mode == "classification" ? x4 = sigmoid(x4) : x4 = x4; } }
	else { x4 = softmaxoverflow(z4); }
}

template<class T>
void templatenet<T>::comp_gradients(doubleMatrix& dW,
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

template<class T>
void templatenet<T>::comp_delta_init(doubleVector& delta,
	const doubleVector& z,
	const doubleVector& x,
	const doubleVector& y) {
	for (unsigned int i = 0; i < delta.size(); ++i) {
		if (n_outputs == 1) {
			mode == "classification" ? delta[i] = sigmoid_prime((z[i]) * (x[i] - y[i]), true) : delta[i] = (z[i]) * (x[i] - y[i]);}
		else { delta[i] = softmax_prime_single(i); }
	}
}

template<class T>
void templatenet<T>::comp_delta(const doubleMatrix& W,
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

template<class T>
void templatenet<T>::backpropagation() {
	comp_delta_init(delta4, z4, x4, y);
	comp_gradients(dW4, db4, x3, delta4);

	comp_delta(W4, z3, delta4, delta3);
	comp_gradients(dW3, db3, x2, delta3);

	comp_delta(W3, z2, delta3, delta2);
	comp_gradients(dW2, db2, x1, delta2);

	comp_delta(W2, z1, delta2, delta1);
	comp_gradients(dW1, db1, x, delta1);
}

template<class T>
void templatenet<T>::gradient_descent() {
	descent(W4, b4, dW4, db4);
	descent(W3, b3, dW3, db3);
	descent(W2, b2, dW2, db2);
	descent(W1, b1, dW1, db1);
}
template<class T>
void templatenet<T>::descent(doubleMatrix& W,
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


template<class T>
double templatenet<T>::loss_function_cross_entropy(double epsilon) {
	doubleVector loss_vec;
	transform(y.begin(), y.end(), x4.begin(), std::back_inserter(loss_vec), [&](double x, double y) {return x * std::log(y + epsilon); });
	double loss = std::accumulate(loss_vec.begin(), loss_vec.end(), 0.0);
	return -loss;
}

template<class T>
double templatenet<T>::comp_loss_mse() {
	double loss = 0.0;
	for (unsigned int i = 0; i < y.size(); ++i) {
		loss += std::pow(y[i] - x4[i], 2);
	}
	return loss;
}

template<class T>
double templatenet<T>::comp_accuracy() {
	double accuracy = 0.0;
	unsigned int prediction = std::distance(x4.begin(), std::max_element(x4.begin(), x4.end()));
	unsigned int ground_truth = std::distance(y.begin(), std::max_element(y.begin(), y.end()));
	if (prediction == ground_truth) {
		accuracy += 1.0;
	}
	return accuracy;
}

template<class T>
void templatenet<T>::comp_stats(const actualType& data, const intVector& labels) {
	double loss = 0.0;
	double accuracy = 0.0;
	for (unsigned int i = 0; i < data.size(); ++i) {
		std::fill(y.begin(), y.end(), 0);
			for (unsigned int k = 0; k < data[0].size(); ++k) {
				x[k] = static_cast<double>(data[i][k]);
			}
		if (n_outputs == 1) {
			y[0] = static_cast<double>(labels[i]);
		}
		else {
			y[labels[i]] = 1.0;
		}
		feedforward();
		if (n_outputs == 1) {
			loss += comp_loss_mse();
		}
		else {
			double epsilon = 1e-8;
			loss += loss_function_cross_entropy(epsilon);
		}
		accuracy += comp_accuracy();
	}
	loss /= static_cast<double>(data.size());
	accuracy /= static_cast<double>(data.size());
	std::cout << "loss is: " << loss << " accuracy is: " << accuracy << std::endl;
}


template<class T>
doubleVector templatenet<T>::comp_prediction(const actualType& preds) {
	doubleVector prediction(preds.size());
	for (unsigned int i = 0; i < preds.size(); ++i) {
		x[i] = static_cast<double>(preds[i]);
		feedforward();
		prediction[i] = x4[0];
	}
	return prediction;
}

template<class T>
doubleVector templatenet<T>::convert_probs_to_class(doubleVector & probs) {
	doubleVector::iterator result = max_element(probs.begin(), probs.end());
	int argmaxVal = distance(probs.begin(), result);
	int selected_class = static_cast<int>(probs[argmaxVal]);
	doubleVector one_hot_classes(probs.size());
	fill(one_hot_classes.begin(), one_hot_classes.end(), 0.0);
	one_hot_classes[selected_class] = 1.0;
	return one_hot_classes;
}

template<class T>
doubleVector templatenet<T>::softmaxoverflow(doubleVector & weights) {
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


template<class T>
doubleVector templatenet<T>::matmul(const doubleMatrix& W, const doubleVector& x, const doubleVector& b)
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

template<class T>
doubleVector templatenet<T>::relu(const doubleVector& z) {
	doubleVector x(z.size());
	for (unsigned int i = 0; i < z.size(); ++i) {
		x[i] = (z[i] >= 0.0 ? z[i] : 0.0);
	};
	return x;
}
template<class T>
double templatenet<T>::relu_prime(const double z) {
	return (z >= 0.0 ? 1.0 : 0.0);
}

template<class T>
double templatenet<T>::sigmoid_prime(const double z, bool first) {
	const double denominator = (1.0 + std::exp(-z));
	const double denominator2 = (1.0 + std::exp(z));
	double sigma;
	sigma = (z > 0.0) ? (1.0 / denominator) : (std::exp(z) / denominator2);
	if (first == true) {
		{ return sigma * (1.0 - sigma); };

	};
	return sigma;
}

template<class T>
doubleVector templatenet<T>::sigmoid(const doubleVector& z) {
	doubleVector x(z.size());
	for (unsigned int i = 0; i < z.size(); ++i) {
		x[i] = sigmoid_prime(z[i], false);
	}
	return x;
}

template<class T>
inline double templatenet<T>::softmax_prime_single(unsigned int index) {
	double result;
	result = x4[index] - y[index];
	return result;
}

template<class T>
double templatenet<T>::softmax_prime() {
	double result;
	for (unsigned int i = 0; i < y.size(); ++i) {
		result += x4[i] - y[i];
	}
	return result;
};
