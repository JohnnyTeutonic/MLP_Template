#pragma once
#include <vector>
struct Point {
	double x;
	double y;
	unsigned int label;
};

struct Point2D {
	int x;
	int y;
	unsigned int sz;
	Point2D::Point2D(unsigned int size) : sz(size) {};
};

using pointVector = std::vector<Point>;

using doubleVector = std::vector<double>;
using doubleMatrix = std::vector<std::vector<double>>;
using intVector = std::vector<int>;

