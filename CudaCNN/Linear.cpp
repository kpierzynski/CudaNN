#include <iostream>

#include "Linear.h"

Linear::Linear(int input_size, int output_size) : Layer(input_size, output_size), input(1,1) {
	weights = new Matrix(input_size, output_size);
	bias = new Matrix(1, output_size);

}

Matrix& Linear::forward(Matrix& input) {
	//std::cout << "LINEAR input" << input.getColumns() << " " << input.getRows() << std::endl;
	//std::cout << "LINEAR weights" << weights->getColumns() << " " << weights->getRows() << std::endl;
	this->input = input;

	Matrix * output = new Matrix(input.getColumns(), weights->getRows());
	*output = ((input) * (*weights)) + (*bias);

	return *output;
}

Matrix& Linear::backward(Matrix& input, float lr) {

	Matrix* input_errors = new Matrix(input.getColumns(), weights->getColumns());
	*input_errors = input * weights->transpose();

	//std::cout << "this->input" << this->input.getColumns() << " " << this->input.getRows() << std::endl;
	//std::cout << "input" << input.getColumns() << " " << input.getRows() << std::endl;
	Matrix weights_errors = (this->input).transpose() * input;

	(*weights) -= weights_errors * lr;
	(*bias) -= input * lr;

	return *input_errors;
}