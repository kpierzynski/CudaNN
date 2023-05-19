#pragma once
#include <cstdint>
#include <random>

class Layer {
public:
	int input_size;
	int output_size;

	Layer(int input_size, int output_size);

	virtual std::vector<float> forward(std::vector<float>& data) = 0;
	virtual std::vector<float> backward(std::vector<float>& gradient, float lr) = 0;

};