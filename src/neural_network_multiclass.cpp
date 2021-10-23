#include "../include/neural_network_multiclass.h"
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <numeric>
#include <stdlib.h>

#include "../include/utils.hpp"
#include "../include/neural_network_updated.hpp"
#include "../include/random_index_generator.hpp"


neural_network_multiclass::neural_network_multiclass(unsigned int _n_inputs,
	unsigned int _n_hidden_1,
	unsigned int _n_hidden_2,
	unsigned int _n_hidden_3,
	unsigned int _n_outputs,
	unsigned int _n_epochs,
	double _learning_rate) : gen{ std::random_device()() } {

	// Initialize network 
	n_inputs = _n_inputs;
	n_hidden_1 = _n_hidden_1;
	n_hidden_2 = _n_hidden_2;
	n_hidden_3 = _n_hidden_3;
	n_outputs = _n_outputs;
	n_epochs = _n_epochs;
	learning_rate = _learning_rate;

	// Weight initialization
	W1.resize(n_hidden_1, std::vector<double>(n_inputs, 0.0));
	W2.resize(n_hidden_2, std::vector<double>(n_hidden_1, 0.0));
	W3.resize(n_hidden_3, std::vector<double>(n_hidden_2, 0.0));
	W4.resize(n_outputs, std::vector<double>(n_hidden_3, 0.0));
	he_initialization(W1);
	he_initialization(W2);
	he_initialization(W3);
	he_initialization(W4);

	// Bias initialization
	b1.resize(n_hidden_1, 0.0);
	b2.resize(n_hidden_2, 0.0);
	b3.resize(n_hidden_3, 0.0);
	b4.resize(n_outputs, 0.0);

	// Cached parameters
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

	dW1.resize(n_hidden_1, std::vector<double>(n_inputs, 0.0));
	dW2.resize(n_hidden_2, std::vector<double>(n_hidden_1, 0.0));
	dW3.resize(n_hidden_3, std::vector<double>(n_hidden_2, 0.0));
	dW4.resize(n_outputs, std::vector<double>(n_hidden_3, 0.0));

	db1.resize(n_hidden_1, 0.0);
	db2.resize(n_hidden_2, 0.0);
	db3.resize(n_hidden_3, 0.0);
	db4.resize(n_outputs, 0.0);
}

void neural_network_multiclass::he_initialization(doubleMatrix& W) {
	double mean = 0.0;
	double std = std::sqrt(2.0 / static_cast<double>(W.size()));
	std::normal_distribution<double> rand_normal(mean, std);
	for (unsigned int i = 0; i < W.size(); ++i) {
		for (unsigned int j = 0; j < W[0].size(); ++j) {
			W[i][j] = rand_normal(gen);
		}
	}
}

void neural_network_multiclass::run(pointVector data_train, pointVector data_valid) {
	RandomIndex rand_idx(data_train.size());
	unsigned int idx;

	for (unsigned int i = 0; i < n_epochs; ++i) {
		std::cout << "epoch no. " << i << '\n';
		for (unsigned int j = 0; j < data_train.size(); ++j) {
			idx = rand_idx.get();
			x[0] = data_train[idx].x;
			x[1] = data_train[idx].y;
			std::fill(y.begin(), y.end(), 0.0);
			y[data_train[idx].label] = 1.0;

			feedforward();
			backpropagation();
			gradient_descent();
		}

		if (i % 2 == 0) {
			comp_stats(data_valid);
		}

		if (i % 50 == 0) {
			comp_prediction_landscape();
		}
	}
}

void neural_network_multiclass::feedforward() {
	z1 = matmul(W1, x, b1);
	x1 = relu(z1);
	z2 = matmul(W2, x1, b2);
	x2 = relu(z2);
	z3 = matmul(W3, x2, b3);
	x3 = relu(z3);
	z4 = matmul(W4, x3, b4);
	x4 = softmaxoverflow(z4);
}

void neural_network_multiclass::comp_gradients(doubleMatrix& dW,
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

void neural_network_multiclass::comp_delta_init(doubleVector& delta,
	const doubleVector& z,
	const doubleVector& x,
	const doubleVector& y) {
	for (unsigned int i = 0; i < delta.size(); ++i) {
		delta[i] = softmax_gradient((z[i]) * (x[i] - y[i]));
	}
}


void neural_network_multiclass::comp_delta(const doubleMatrix& W,
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


void neural_network_multiclass::backpropagation() {
	comp_delta_init(delta4, z4, x4, y);
	comp_gradients(dW4, db4, x3, delta4);

	comp_delta(W4, z3, delta4, delta3);
	comp_gradients(dW3, db3, x2, delta3);

	comp_delta(W3, z2, delta3, delta2);
	comp_gradients(dW2, db2, x1, delta2);

	comp_delta(W2, z1, delta2, delta1);
	comp_gradients(dW1, db1, x, delta1);
}

void neural_network_multiclass::gradient_descent() {
	descent(W4, b4, dW4, db4);
	descent(W3, b3, dW3, db3);
	descent(W2, b2, dW2, db2);
	descent(W1, b1, dW1, db1);
}

void neural_network_multiclass::descent(doubleMatrix& W,
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

double neural_network_multiclass::comp_loss() {
	double loss = 0.0;
	for (unsigned int i = 0; i < y.size(); ++i) {
		loss += std::pow(y[i] - x4[i], 2);
	}
	return loss;
}


double neural_network_multiclass::loss_function_cross_entropy(double epsilon = 1e-8) {
	doubleVector loss_vec;
	transform(y.begin(), y.end(), x4.begin(), std::back_inserter(loss_vec), [&](double x, double y) {return x * log(y + epsilon); });
	double loss = std::accumulate(loss_vec.begin(), loss_vec.end(), 0.0f);
	return -loss;
}


double neural_network_multiclass::comp_accuracy() {
	double accuracy = 0.0;
	unsigned int prediction = std::distance(x4.begin(), std::max_element(x4.begin(), x4.end()));
	unsigned int ground_truth = std::distance(y.begin(), std::max_element(y.begin(), y.end()));
	if (prediction == ground_truth) {
		accuracy += 1.0;
	}
	return accuracy;
}

void neural_network_multiclass::comp_stats(const pointVector& data) {
	double loss = 0.0;
	double accuracy = 0.0;
	for (unsigned int i = 0; i < data.size(); ++i) {
		std::fill(y.begin(), y.end(), 0.0);
		x[0] = data[i].x;
		x[1] = data[i].y;
		y[data[i].label] = 1.0;
		feedforward();
		loss += loss_function_cross_entropy();
		accuracy += comp_accuracy();
	}
	loss /= static_cast<double>(data.size());
	accuracy /= static_cast<double>(data.size());
	std::cout << "loss is: " << loss << " accuracy is: " << accuracy << std::endl;
}

pointVector neural_network_multiclass::comp_grid(const unsigned int n_points_x,
	const unsigned int n_points_y,
	const double x_min,
	const double y_min,
	const double x_max,
	const double y_max) {
	const double dx = (x_max - x_min) / static_cast<double>(n_points_x - 1);
	const double dy = (y_max - y_min) / static_cast<double>(n_points_y - 1);

	pointVector grid(n_points_x * n_points_y);
	double pos_x = x_min;
	double pos_y = y_max;
	unsigned int idx = 0;

	for (unsigned int i = 0; i < n_points_y; ++i) {
		for (unsigned int j = 0; j < n_points_x; ++j) {
			grid[idx].x = pos_x;
			grid[idx].y = pos_y;
			pos_x += dx;
			++idx;
		}
		pos_x = x_min;
		pos_y -= dy;
	}
	return grid;
}

doubleVector neural_network_multiclass::comp_prediction(const pointVector& grid) {
	doubleVector prediction(grid.size());
	for (unsigned int i = 0; i < grid.size(); ++i) {
		x[0] = grid[i].x;
		x[1] = grid[i].y;
		feedforward();
		prediction[i] = x4[0];
	}
	return prediction;
}


doubleVector convert_probs_to_class(doubleVector & probs) {
	doubleVector::iterator result = max_element(probs.begin(), probs.end());
	int argmaxVal = distance(probs.begin(), result);
	int selected_class = probs[argmaxVal];
	doubleVector one_hot_classes(probs.size());
	fill(one_hot_classes.begin(), one_hot_classes.end(), 0.0);
	one_hot_classes[selected_class] = 1.0;
	return one_hot_classes;
}



void neural_network_multiclass::calculate_gradients(doubleVector& delta, const doubleVector& z,const doubleVector& x, const doubleVector& y) {
	size_t count = delta.size();
	const doubleVector softmaxResult = z4;
	const doubleVector gradFromAbove = x4;
	doubleVector gradOutput;

	//delta4, z4, x4, y
	SoftmaxGrad_fromResult_nonSSE1(z4, x4, gradOutput, count);

}

inline void neural_network_multiclass::SoftmaxGrad_fromResult_nonSSE1(doubleVector softmaxResult,
	const doubleVector gradFromAbove,  //<--gradient vector, flowing into us from the above layer
	doubleVector gradOutput,
	size_t count) {
	// every pre-softmax element in a layer contributed to the softmax of every other element
	// (it went into the denominator). So gradient will be distributed from every post-softmax element to every pre-elem.
	for (size_t i = 0; i < count; ++i) {
		//go through this i'th row:
		float sum = 0.0f;

		const float neg_sft_i = -softmaxResult[i];

		for (size_t j = 0; j < count; ++j) {
			float mul = gradFromAbove[j] * softmaxResult[j] * neg_sft_i;
			sum += mul;//adding to the total sum of this row.
		}
		//NOTICE: equals, overwriting any old values:
		gradOutput[i] = sum;
	}//end for every row

	for (size_t i = 0; i < count; ++i) {
		gradOutput[i] += softmaxResult[i] * gradFromAbove[i];
	}
}

double neural_network_multiclass::softmax_gradient(double z) {
	const double denominator = (1.0 + std::exp(-z));
	const double denominator2 = (1.0 + std::exp(z));
	double sigma;
	sigma = (z > 0.0) ? (1.0 / denominator) : (std::exp(z) / denominator2);
	return sigma * (1.0 - sigma); 
};

	//doubleVector act = softmaxoverflow(z);
	//for (int i = 0; i < act.size(); i++) {
	//	derivativeWeights.emplace_back(act[i] * (1.0 - act[i]));
	//}
	//return derivativeWeights;


doubleVector neural_network_multiclass::softmaxoverflow(doubleVector & weights) {
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



void neural_network_multiclass::write_pred_to_file(const doubleVector pred,
	const unsigned int n_points_x,
	const unsigned int n_points_y) {

	std::string file_name = "prediction_landscape.dat";
	std::ofstream file;
	file.open(file_name);

	if (file.fail()) {
		std::cerr << "Error\n";
	}
	else {
		unsigned int idx = 0;
		for (unsigned int i = 0; i < n_points_y; ++i) {
			for (unsigned int j = 0; j < n_points_x; ++j) {
				file << pred[idx] << " ";
				++idx;
			}
			file << '\n';
		}
		file.close();
	}
}

void neural_network_multiclass::comp_prediction_landscape() {
	const unsigned int n_points_x = 256;
	const unsigned int n_points_y = 256;
	const double x_min = -1.0;
	const double y_min = -1.0;
	const double x_max = 1.0;
	const double y_max = 1.0;

	pointVector grid = comp_grid(n_points_x, n_points_y, x_min, y_min, x_max, y_max);
	doubleVector pred = comp_prediction(grid);
	write_pred_to_file(pred, n_points_x, n_points_y);
}

doubleVector neural_network_multiclass::matmul(const doubleMatrix& W,
	const doubleVector& x,
	const doubleVector& b) {
	doubleVector z(W.size(), 0.0);
	for (unsigned int i = 0; i < W.size(); ++i) {
		for (unsigned int j = 0; j < W[0].size(); ++j) {
			z[i] += W[i][j] * x[j];
		}
		z[i] += b[i];
	}
	return z;
}

doubleVector neural_network_multiclass::relu(const doubleVector& z) {
	doubleVector x(z.size());
	for (unsigned int i = 0; i < z.size(); ++i) {
		x[i] = (z[i] >= 0.0 ? z[i] : 0.0);
	};
	return x;
}

double neural_network_multiclass::relu_prime(const double z) {
	return (z >= 0.0 ? 1.0 : 0.0);
}


double neural_network_multiclass::sigmoid_prime(const double z, bool first = true) {
	const double denominator = (1.0 + std::exp(-z));
	double sigma;
	sigma = (z > 0.0) ? (1.0 / denominator) : (std::exp(z) / denominator);
	if (first == true) {
		{ return sigma * (1.0 - sigma); };

	};
	return sigma;
}


doubleVector neural_network_multiclass::sigmoid(const doubleVector& z) {
	doubleVector x(z.size());
	for (unsigned int i = 0; i < z.size(); ++i) {
		x[i] = sigmoid_prime(z[i], false);
	}
	return x;
};

