#include <iostream>

#include "Linear.h"

Linear::Linear(int input_size, int output_size) : Layer(input_size, output_size) {
	weights = new Matrix(input_size, output_size);
	bias = new Matrix(1, output_size);
}

std::vector<float> Linear::forward(std::vector<float>& data) {
	input = new Matrix(1, input_size, data);

	Matrix output = ((*input) * (*weights)) + (*bias);

	return output.data;
}

std::vector<float> Linear::backward(std::vector<float>& gradient, float lr) {
	Matrix m_gradient = Matrix(1, output_size, gradient);
	Matrix input_errors = m_gradient * weights->transpose();

	Matrix weights_errors = (*input).transpose() * m_gradient;

	(*weights) -= weights_errors * lr;
	(*bias) -= m_gradient * lr;

	return input_errors.data;
}