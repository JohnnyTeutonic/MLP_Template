#include <random>

class RandomNumberBetween
{
public:
	RandomNumberBetween(int low, int high);
	int operator()();
private:
	std::mt19937 random_engine_;
	std::uniform_int_distribution<int> distribution_;
};
