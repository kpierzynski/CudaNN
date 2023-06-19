#include "Tanh.h"

#include <cmath>

__global__ void tanhForward(float* input_data, float* output_data, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		output_data[index] = tanh(input_data[index]);
	}
}

__global__ void tanhBackward(float* input_data, float* output_data, float* grad, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		float tanh_val = tanh(input_data[index]);
		output_data[index] = (1.0f - tanh_val * tanh_val) * grad[index];
	}
}

Tanh::Tanh(int size, int batch_size) : Layer(size, size), batch_size(batch_size) {
	input = new Tensor(batch_size, input_size);

	dA = new Tensor(batch_size, input_size);
	output = new Tensor(batch_size, output_size);
}

Tensor* Tanh::forward(Tensor& input) {
	delete this->input;

	this->input = new Tensor(input);

	int size = input.rows * input.cols;
	int blockSize = size;
	int gridSize = (size + blockSize) / blockSize;


	tanhForward << <gridSize, blockSize >> > (input.dev, output->dev, input_size*batch_size);
	cudaDeviceSynchronize();

	return output;
}

Tensor* Tanh::backward(Tensor& input, float lr) {
	int block_size = 256;
	int grid_size = (input_size*batch_size + block_size - 1) / block_size;
	tanhBackward << <grid_size, block_size >> > (this->input->dev, dA->dev, input.dev, input_size*batch_size);
	cudaDeviceSynchronize();
	return dA;
}