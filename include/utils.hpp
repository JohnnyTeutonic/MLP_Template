#pragma once
#include <utility>
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
	Point2D(unsigned int size) : sz(size) {};
};


struct Point3D {
	int x;
	int y;
	int z;
	unsigned int sz;
	Point3D(unsigned int size) : sz(size) {};
};


struct Point4D {
	int w;
	int x;
	int y;
	int z;
	unsigned int sz;
	Point4D(unsigned int size) : sz(size) {};
};

using pointVector = std::vector<Point>;

using doubleVector = std::vector<double>;
using doubleMatrix = std::vector<std::vector<double>>;
using intVector = std::vector<int>;

template <class T, class M> M get_member_type(M T:: *);
#define GET_TYPE_OF(mem) decltype(get_member_type(mem))
/*
example usage:
using elemType = typename std::decay<decltype(*data_test_mat.begin())>::type;
elemType mytype(4);
GET_TYPE_OF(&elemType::x);
*/
template<class T, class M>
inline M get_member_type(M T::*)
{
	return M();
}

template< typename Callable, typename... Arguments > void
Ignore_Exceptions(Callable && method, Arguments && ... arguments) noexcept
{
	try
	{
		method(::std::forward< Arguments >(arguments)...);
	}
	catch (...)
	{
	}
}

