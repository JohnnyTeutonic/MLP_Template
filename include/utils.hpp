#pragma once
#include <vector>
struct Point {
	double x;
	double y;
	unsigned int label;
};

using pointVector = std::vector<Point>;

using doubleVector = std::vector<double>;
using doubleMatrix = std::vector<std::vector<double>>;
using intVector = std::vector<int>;

