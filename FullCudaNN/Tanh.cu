#include "Tanh.h"

#include <cmath>

__global__ void tanhForward(float* input_data, float* output_data, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
		output_data[index] = tanh(input_data[index]);
}

__global__ void tanhBackward(float* input_data, float* output_data, float* gradient, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		float tanh_val = tanh(input_data[index]);
		output_data[index] = (1.0f - tanh_val * tanh_val) * gradient[index];
	}
}

Tanh::Tanh(int size, int batch_size) : Layer(size, size), batch_size(batch_size) {
	gradient = new Tensor(batch_size, input_size);
	output = new Tensor(batch_size, output_size);
}

Tensor* Tanh::forward(Tensor& input) {
	this->input = &input;

	int size = input.rows * input.cols;

	{
		int dimBlock = size;
		int dimGrid = (size + dimBlock) / dimBlock;
		tanhForward << <dimGrid, dimBlock >> > (input.dev, output->dev, input_size * batch_size);
	}

	return output;
}

Tensor* Tanh::backward(Tensor& input, float lr) {
	{
		int dimBlock = 1024;
		int dimGrid = (input_size * batch_size + dimBlock - 1) / dimBlock;
		tanhBackward << <dimGrid, dimBlock >> > (this->input->dev, gradient->dev, input.dev, input_size * batch_size);
	}

	return gradient;
}