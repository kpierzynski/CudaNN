#include "Linear.h"

Linear::Linear(int input_size, int output_size) : 
	Layer(input_size, output_size), 
	weights(Tensor(input_size,output_size)), 
	biases(Tensor(1,output_size))
{

}

Tensor Linear::forward(Tensor& input)
{
	Tensor output = input * weights;

	output += biases;

	return output;
}

Tensor Linear::backward(Tensor& input, float lr)
{
	Tensor inputGradient = weights * input.transpose();
	weights -= (inputGradient * input) * lr;
	biases -= inputGradient * lr;

	return inputGradient;
}

