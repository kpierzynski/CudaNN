#pragma once

#include <iostream>

#include "Tensor.h"
#include "Layer.h"

class Linear : public Layer {
	public:
	Linear(int input_size, int output_size);

	Tensor forward(Tensor& input) override;
	Tensor backward(Tensor& input, float lr) override;

	private:
	Tensor weights;
	Tensor biases;
	Tensor input;

	Tensor output;

	float* dev_output;
	float* dev_input;
	float* dev_weights;
	float* dev_biases;

};