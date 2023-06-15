#include "Linear.h"

#include <cstdio>
#include <iostream>

__global__ void forwardKernel(float* W, float* A, float* Z, float* b,
							  int W_x_dim, int W_y_dim,
							  int A_x_dim, int A_y_dim) {

	//printf("forwardKernel\r\n");

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int Z_x_dim = A_x_dim;
	int Z_y_dim = W_y_dim;

	float Z_value = 0;

	if (row < Z_y_dim && col < Z_x_dim) {
		for (int i = 0; i < W_x_dim; i++) {
			Z_value += W[row * W_x_dim + i] * A[i * A_x_dim + col];
		}
		Z[row * Z_x_dim + col] = Z_value + b[row];
	}

	//printf("Z[%d * %d + %d]: %f, W[row * Z_x_dim + col]: %f\r\n", row, Z_x_dim, col, Z[row * Z_x_dim + col], W[row * Z_x_dim + col]);
}

__global__ void backwardKernel(float* W, float* dZ, float* dA,
							   int W_x_dim, int W_y_dim,
							   int dZ_x_dim, int dZ_y_dim) {

	//printf("backwardKernel\r\n");

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	// W is treated as transposed
	int dA_x_dim = dZ_x_dim;
	int dA_y_dim = W_x_dim;

	float dA_value = 0.0f;

	if (row < dA_y_dim && col < dA_x_dim) {
		for (int i = 0; i < W_y_dim; i++) {
			dA_value += W[i * W_x_dim + row] * dZ[i * dZ_x_dim + col];
		}
		dA[row * dA_x_dim + col] = dA_value;
	}
}

__global__ void updateWeightsKernel(float* dZ, float* A, float* W,
									int dZ_x_dim, int dZ_y_dim,
									int A_x_dim, int A_y_dim,
									float learning_rate) {
	//printf("updateWeightsKernel\r\n");
	//printf("int %d, int %d, int %d, int %d\r\n", dZ_x_dim, dZ_y_dim, A_x_dim, A_y_dim);

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;


	// A is treated as transposed
	int W_x_dim = A_y_dim;
	int W_y_dim = dZ_y_dim;

	float dW_value = 0.0f;

	if (row < W_y_dim && col < W_x_dim) {
		for (int i = 0; i < dZ_x_dim; i++) {
			dW_value += dZ[row * dZ_x_dim + i] * A[col * A_x_dim + i];
		}
		W[row * W_x_dim + col] = W[row * W_x_dim + col] - learning_rate * (dW_value / A_x_dim);
		//printf("row: %d, col: %d\r\n", row, col);
	}
}

__global__ void updateBiasKernel(float* dZ, float* b,
								 int dZ_x_dim, int dZ_y_dim,
								 int b_x_dim,
								 float learning_rate) {

	//printf("updateBiasKernel\r\n");

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < dZ_x_dim * dZ_y_dim) {
		int dZ_x = index % dZ_x_dim;
		int dZ_y = index / dZ_x_dim;
		atomicAdd(&b[dZ_y], - learning_rate * (dZ[dZ_y * dZ_x_dim + dZ_x] / dZ_x_dim));
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

Tensor& Linear::forward(Tensor& input)
{
	this->input = new Tensor(input);

	//std::cout << "FORWARD:this->input " << this->input->rows << " " << this->input->cols << std::endl;
	//std::cout << "FORWARD:input" << input.rows << " " << input.cols << std::endl;

	dim3 block_size(32, 32);
	dim3 num_of_blocks((output->rows + block_size.x-1) / block_size.x,
					   (output->cols + block_size.y-1) / block_size.y);

	forwardKernel << <num_of_blocks, block_size >> > (
		weights->dev,
		this->input->dev,
		output->dev,
		biases->dev,
		weights->rows, weights->cols,
		this->input->rows, this->input->cols
		);

	cudaDeviceSynchronize();

	return *output;
}

Tensor& Linear::backward(Tensor& input, float lr)
{
	//std::cout << "BACKWARD: " << this->input->rows << " " << this->input->cols << std::endl;

	/*	CPU BACKWARD
	Tensor Linear::backward(Tensor& gradient, float lr)
	{
							8 x 784	8x10 - > 784x8 * 8x10 => 784x10
		Tensor wGradient = input.transpose() * gradient;
		Tensor bGradient = gradient;

		weights -= wGradient * lr;
		// sum_rows sumuje wiersze w jeden wiersz
		biases -= (bGradient * lr).sum_rows();

		Tensor output = gradient * weights.transpose();

		return output;
	}
	*/

	//std::cout << "backward input arg dims: " << input.rows << " " << input.cols << std::endl;

	//std::cout << "STEP1 FUNCTION" << std::endl << std::endl;
	dim3 block_size(32, 32);
	dim3 num_of_blocks((this->input->rows + block_size.x - 1) / block_size.x,
					   (this->input->cols + block_size.y - 1) / block_size.y);
	cudaDeviceSynchronize();

	backwardKernel << <num_of_blocks, block_size >> > (
		weights->dev,
		this->input->dev,
		dA->dev,

		weights->rows, weights->cols,
		this->input->rows, this->input->cols
		);

	dim3 num_of_blocks1((input.cols*input.rows + block_size.x-1) / block_size.x);
	/*updateBiasKernel << <num_of_blocks1, block_size >> > (
		input.dev,
		biases->dev,
		input.rows, input.cols,
		biases->rows, lr
		);
	cudaDeviceSynchronize();
	*/

	dim3 num_of_blocks2((weights->rows + block_size.x-1) / block_size.x,
						(weights->cols + block_size.y-1) / block_size.y);


	updateWeightsKernel << <num_of_blocks2, block_size >> > (
		input.dev,
		this->input->dev,
		weights->dev,

		input.rows, input.cols,
		this->input->rows, this->input->cols,
		lr);
	cudaDeviceSynchronize();

	weights->dev2host();
	weights->print();

	biases->dev2host();
	biases->print();

	return *dA;
}
