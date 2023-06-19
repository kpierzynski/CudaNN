#include "Linear.h"

#include <cstdio>
#include <iostream>

__global__ void forwardKernel(float* A, float* B, float* C, float* b, int numARows,
					  int numAColumns, int numBRows, int numBColumns) {


	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < numARows && col < numBColumns) {
		float sum = 0;
		for (int ii = 0; ii < numAColumns; ii++) {
			sum += A[row * numAColumns + ii] * B[ii * numBColumns + col];
		}
		C[row * numBColumns + col] = sum + b[row * numBColumns + col];

	}
}

__global__ void backwardKernel(float* A, float* B, float* C, int rowsA, int colsA, int colsB)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < rowsA && col < colsB)
	{
		float sum = 0.0f;

		for (int i = 0; i < colsA; ++i)
		{
			sum += A[row * colsA + i] * B[col * colsA + i];
		}

		C[row * colsB + col] = sum;
	}
}

__global__ void updateWeightsKernel(float* W, const float* A, const float* B, float lr, int rowsW, int colsW, int rowsA, int colsA, int rowsB, int colsB) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < colsA && col < colsB) {
		float value = 0.0f;

		for (int i = 0; i < rowsB; i++) {
			value += A[i * colsA + row] * B[i * colsB + col];
		}

		W[row * colsW + col] -= lr * value;
		//printf("uWK: %d, value: %f\r\n", row * colsW + col, value);
	}
}

__global__ void updateBiasKernel(float* dZ, float* b,
								 int dZ_x_dim, int dZ_y_dim,
								 int b_x_dim,
								 float learning_rate) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < dZ_x_dim && col < dZ_y_dim) {
		atomicAdd(&b[row * dZ_y_dim + col], -learning_rate * dZ[row * dZ_y_dim + col]);
	}
}

Linear::Linear(int input_size, int output_size, int batch_size) : Layer(input_size, output_size)
{
	weights = new Tensor(input_size, output_size);
	biases = new Tensor(batch_size, output_size);
	input = new Tensor(batch_size, input_size);

	dA = new Tensor(batch_size, input_size);
	output = new Tensor(batch_size, output_size);

	biases->zero();
}

Tensor* Linear::forward(Tensor& input)
{
	delete this->input;

	this->input = new Tensor(input);

	dim3 blockDim(32, 32);
	dim3 gridDim(ceil(((float)this->input->cols) / blockDim.x),
				 ceil(((float)weights->rows) / blockDim.y));
	forwardKernel << <gridDim, blockDim >> > (this->input->dev, weights->dev, output->dev, biases->dev, this->input->rows, this->input->cols, weights->rows,
									  weights->cols);

	return output;
}

Tensor* Linear::backward(Tensor& input, float lr)
{
	dim3 blockSize(32, 32);

	dim3 gridSize((weights->cols + blockSize.x - 1) / blockSize.x, (input.rows + blockSize.y - 1) / blockSize.y);
	backwardKernel << <gridSize, blockSize >> > (input.dev, weights->dev, dA->dev, input.rows, input.cols, weights->rows);


	dim3 gridDim((weights->cols + blockSize.x - 1) / blockSize.x, (weights->rows + blockSize.y - 1) / blockSize.y);
	updateWeightsKernel << <gridDim, blockSize >> > (weights->dev, this->input->dev, input.dev, lr,
												 weights->rows, weights->cols, this->input->rows, this->input->cols, input.rows, input.cols);

	dim3 num_of_blocks1((input.cols * input.rows + blockSize.x) / blockSize.x);
	updateBiasKernel << <num_of_blocks1, blockSize >> > (
		input.dev,
		biases->dev,
		input.rows, input.cols,
		biases->rows, lr
		);

	return dA;
}
