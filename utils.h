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

using Vec = vector<float>;
using myvariant = variant<float, Vec>;
template <typename It> // templated variant for softmax activation function for different types of iterators
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

struct Node { // for hidden layers and final layers
	Node() = default;
	explicit Node(unsigned int size) : weights(size) { sz = size; };
	Node(const Node& copynode) : weights(copynode.weights) { sz = copynode.sz; };
	unsigned int sz;
	vector<float> weights;
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

struct inputNode { // for the input layer
	inputNode(float val) : value(val) {};
	float value;
	friend ostream& operator <<(ostream& os, const inputNode nd) {
		return os << "[" << nd.value << "]";
	}
};

struct container { // contains the output from forward prop and is carried forward for backprop
	container() = default;
	container(Vec t, Vec u, Vec v, Vec w, Vec x, Vec z) : W1(t), W2(u), A1(v), A2(w), Z1(x), Z2(z) {};
	Vec W1, W2, A1, A2, Z1, Z2;
};

struct Visitor
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


template <typename T>
bool constexpr IsInBounds(const T& value, const T& low, const T& high) {
	return !(value < low) && (value < high);
}


bool is_float_eq(float a, float b, float epsilon) {
	return ((a - b) < epsilon) && ((b - a) < epsilon);
}

template <class T>
static bool compareVectors(vector<T> a, vector<T> b)
{
	if (a.size() != b.size())
	{
		return false;
	}
	return true;
}
