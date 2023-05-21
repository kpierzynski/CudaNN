#pragma once
#include <cstdint>
#include <random>

#include "Matrix.h"

class Layer {
public:
	int input_size;
	int output_size;

	Layer(int input_size, int output_size);

	virtual Matrix& forward(Matrix& input) = 0;
	virtual Matrix& backward(Matrix& input, float lr) = 0;

};