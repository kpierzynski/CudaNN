#pragma once

#include <iostream>

#include "Tensor.h"

class Layer {

	private:
	int input_size;
	int output_size;

	public:
	Layer(int input_size, int output_size);

	virtual Tensor forward(Tensor& input) = 0;
	virtual Tensor backward(Tensor& input, float lr) = 0;

};
