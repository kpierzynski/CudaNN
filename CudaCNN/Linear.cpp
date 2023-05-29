#include "Linear.h"

Linear::Linear(int input_size, int output_size) :
	Layer(input_size, output_size),
	weights(Tensor(input_size, output_size)),
	biases(Tensor(1, output_size)),
	input(Tensor(1, input_size))
{
	weights.set_random();
	biases.set_random();
}

Tensor Linear::forward(Tensor& input)
{
	this->input = input;
	Tensor output = input * weights;
	output += biases;

	return output;
}

Tensor Linear::backward(Tensor& gradient, float lr)
{
	Tensor wGradient = input.transpose() * gradient;
	Tensor bGradient = gradient;

	weights -= wGradient * lr;
	biases -= (bGradient * lr).sum_rows();

	Tensor output = gradient * weights.transpose();

	return output;
}

