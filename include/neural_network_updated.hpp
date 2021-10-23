#pragma once

#include <ostream>
#include <vector>
#include <string>
#include <random>
#include <cstdlib>
#include <ctime>
#include "utils.hpp"

class NeuralNetwork {

public:
	NeuralNetwork(unsigned int _n_inputs,
		unsigned int _n_hidden_1,
		unsigned int _n_hidden_2,
		unsigned int _n_hidden_3,
		unsigned int _n_outputs,
		unsigned int _n_epochs,
		double _learning_rate);

	void run(pointVector data_train, pointVector data_valid);
	std::mt19937 gen;
	unsigned int n_epochs;
	unsigned int n_inputs;
	unsigned int n_hidden_1;
	unsigned int n_hidden_2;
	unsigned int n_hidden_3;
	unsigned int n_outputs;
	double learning_rate;

private:
	doubleMatrix W1;
	doubleMatrix W2;
	doubleMatrix W3;
	doubleMatrix W4;

	doubleVector b1;
	doubleVector b2;
	doubleVector b3;
	doubleVector b4;

	doubleMatrix dW1;
	doubleMatrix dW2;
	doubleMatrix dW3;
	doubleMatrix dW4;

	doubleVector db1;
	doubleVector db2;
	doubleVector db3;
	doubleVector db4;

	doubleVector z1;
	doubleVector z2;
	doubleVector z3;
	doubleVector z4;

	doubleVector x;
	doubleVector x1;
	doubleVector x2;
	doubleVector x3;
	doubleVector x4;
	doubleVector y;

	doubleVector delta1;
	doubleVector delta2;
	doubleVector delta3;
	doubleVector delta4;

	void he_initialization(doubleMatrix& weights);
	void feedforward();

	doubleVector matmul(const doubleMatrix& W,
		const doubleVector& x,
		const doubleVector& b);

	doubleVector relu(const doubleVector& x);
	doubleVector sigmoid(const doubleVector& x);

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

	void comp_stats(const pointVector& data);
	double comp_loss();
	double comp_accuracy();

	void comp_prediction_landscape();

	pointVector comp_grid(const unsigned int n_points_x,
		const unsigned int n_points_y,
		const double x_min,
		const double y_min,
		const double x_max,
		const double y_max);

	doubleVector comp_prediction(const pointVector& grid);

	void write_pred_to_file(const doubleVector pred,
		const unsigned int n_points_x,
		const unsigned int n_points_y);
};
