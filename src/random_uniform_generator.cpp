#include <random>

class RandomNumberBetween
{
public:
	RandomNumberBetween(int low, int high)
		: random_engine_{ std::random_device{}() }
		, distribution_{ low, high }
	{
	}	
	int operator()()
	{
		return static_cast<unsigned int>(distribution_(random_engine_));
	}
private:
	std::mt19937 random_engine_;
	std::uniform_int_distribution<int> distribution_;
};

