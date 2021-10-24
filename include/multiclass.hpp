#pragma once
#include <ostream>
#include <vector>
#include <string>
#include <random>
#include <cstdlib>
#include <ctime>
#include "utils.hpp"

class experimental {

public:
	experimental(unsigned int _n_inputs,
		unsigned int _n_hidden_1,
		unsigned int _n_hidden_2,
		unsigned int _n_hidden_3,
		unsigned int _n_outputs,
		unsigned int _n_epochs,
		double _learning_rate);
	
	void run(intVector data_train, intVector data_valid, intVector class_labels, intVector valid_labels);
	std::mt19937 gen;
	unsigned int n_epochs;
	unsigned int n_inputs;
	unsigned int n_hidden_1;
	unsigned int n_hidden_2;
	unsigned int n_hidden_3;
	unsigned int n_outputs;
	double learning_rate;

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

	void comp_stats(const intVector& data, const intVector& labels);
	double comp_accuracy();
	doubleVector comp_prediction(const intVector& preds);
	doubleVector softmaxoverflow(doubleVector & weights);
	doubleVector convert_probs_to_class(doubleVector & probs);
	double loss_function_cross_entropy(double epsilon);
};
