#pragma once
#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#include <cassert>
#include <ctime>
#include <cstdlib>
using namespace std;

using Vec = vector<double>;
using myvariant = variant<double, Vec>;
template <typename It> // template for softmax activation function for different types of iterators
void softmax(It beg, It end)
{
	using VType = typename iterator_traits<It>::value_type;

	static_assert(is_floating_point<VType>::value,
		"Softmax function only applicable for floating point types");

	auto max_ele{ *max_element(beg, end) };

	transform(beg, end, beg, [&](VType x) { return exp(x - max_ele); });

	VType exptot = accumulate(beg, end, 0.0);

	transform(beg, end, beg, [&](VType x) { auto ex = exp(x - max_ele); exptot += ex; return ex; });
}


struct Node { // holds nodes for the hidden layers and the final layer
	Node() = default;
	~Node() = default;
	explicit Node(unsigned int size) : weights(size) { sz = size; };
	Node(const Node& copynode) : weights(copynode.weights) { sz = copynode.sz; };
	unsigned int sz;
	Vec weights;
	friend ostream& operator <<(ostream& os, const Node nd) {
		os << "[";
		unsigned int last = nd.sz - 1;
		for (unsigned int i = 0; i < nd.sz; ++i) {
			os << nd.weights[i];
			if (i != last)
				os << ", ";
		}
		os << "]";
		return os;
	}
};

struct inputNode { // holds nodes for the input layer
	inputNode(double val) : value(val) {};
	~inputNode() = default;
	double value;
	friend ostream& operator <<(ostream& os, const inputNode nd) {
		return os << "[" << nd.value << "]";
	}
};

struct container { // contains the output from forward prop and is carried forward for backprop
	container() = default;
	container(vector<vector<double>> t, vector<vector<double>> u, Vec v, Vec w, Vec x, Vec z) : W1(t), W2(u), A1(v), A2(w), Z1(x), Z2(z) {};
	Vec A1, A2, Z1, Z2;
	vector<vector<double>> W1, W2;
};

struct Visitor // for use with std::variant
{
	const int operator()(const int & t) const
	{
		return t;
	}
	const Vec operator()(const Vec & V) const
	{
		return V;
	}
};


template <typename T> // ensures the learning rate is bounded
bool constexpr IsInBounds(const T& value, const T& low, const T& high) {
	return !(value < low) && (value < high);
}


bool is_double_eq(double a, double b, double epsilon) { // compares doubleing point values; uses epsilon to ensure the values are approximately equal up to a limit
	return ((a - b) < epsilon) && ((b - a) < epsilon);
}

template <class T> // compares size of vectors
static bool compareVectors(vector<T> a, vector<T> b)
{
	if (a.size() != b.size())
	{
		return false;
	}
	return true;
}

template<typename T> // generic function that multiplies a vector by a vector using arbitrary vector types
auto linear_forward(vector<T> prev, vector<T> next, int index) {
	return inner_product(begin(prev[index].weights), end(prev[index].weights), begin(next[index].weights), 0.0);
}

Vec double_dot(const vector<Node> & W, const Vec & x) { // matrix-vector product - results in a vector
	Vec z(W.size(), 0.0);
	for (unsigned int i = 0; i < W.size(); ++i) {
		for (unsigned int j = 0; j < W[0].weights.size(); ++j) {
			z[i] += W[i].weights[j] * x[j];
		}
	}
	return z;
}



Vec matmul(const vector<Node> & W, const Vec & x, const Vec & b) { // matrix-vector product - results in a vector
	Vec z(W.size(), 0.0);
	for (unsigned int i = 0; i < W.size(); ++i) {
		for (unsigned int j = 0; j < W[0].weights.size(); ++j) {
			z[i] += W[i].weights[j] * x[j];
		}
		z[i] += b[i];
	}
	return z;
}

double matmul_no_bias(const Vec & W, const Vec & x) { // matrix-vector without bias term product - results in a vector
	//Vec z(W.size(), 0.0);
	double z = 0.0f;
	for (unsigned int i = 0; i < W.size(); ++i) {
		for (unsigned int j = 0; j < x.size(); ++j) {
			z += W[j] * x[j];
		}
	}
	return z;
}

