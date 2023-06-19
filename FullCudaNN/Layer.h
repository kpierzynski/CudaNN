#pragma once

#include "Tensor.h"

class Layer {

	public:
	Layer(int input_size, int output_size);

	virtual Tensor * forward(Tensor& input) = 0;
	virtual Tensor * backward(Tensor& input, float lr) = 0;

	protected:
	int input_size;
	int output_size;
};