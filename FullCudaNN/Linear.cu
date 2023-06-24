#include "Linear.h"

#include <cstdio>
#include <iostream>
#include <chrono>

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void forwardKernel(float* A, float* B, float* C, float* b, int numARows,
							  int numAColumns, int numBRows, int numBColumns) {
	//@@ Insert code to implement matrix multiplication here
	__shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y,
		Row = by * TILE_WIDTH + ty, Col = bx * TILE_WIDTH + tx;
	float Pvalue = 0;

	for (int m = 0; m < (numAColumns - 1) / TILE_WIDTH + 1; ++m) {
		if (Row < numARows && m * TILE_WIDTH + tx < numAColumns)
			ds_M[ty][tx] = A[Row * numAColumns + m * TILE_WIDTH + tx];
		else
			ds_M[ty][tx] = 0;
		if (Col < numBColumns && m * TILE_WIDTH + ty < numBRows)
			ds_N[ty][tx] = B[(m * TILE_WIDTH + ty) * numBColumns + Col];
		else
			ds_N[ty][tx] = 0;

		__syncthreads();
		for (int k = 0; k < TILE_WIDTH; ++k)
			Pvalue += ds_M[ty][k] * ds_N[k][tx];
		__syncthreads();
	}
	if (Row < numARows && Col < numBColumns)
		C[Row * numBColumns + Col] = Pvalue;
}

__global__ void forwardKernel2(float* A, float* B, float* C, float* b, int numARows,
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
	}
}


__global__ void updateBiasKernel(float* dZ, float* b,
								 int dZ_x_dim, int dZ_y_dim,
								 int b_x_dim,
								 float learning_rate) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < dZ_x_dim && col < dZ_y_dim) {
		b[row * dZ_y_dim + col] -= learning_rate * dZ[row * dZ_y_dim + col];
	}
}

Linear::Linear(int input_size, int output_size, int batch_size) : Layer(input_size, output_size)
{
	weights = new Tensor(input_size, output_size);
	biases = new Tensor(batch_size, output_size);

	dA = new Tensor(batch_size, input_size);
	output = new Tensor(batch_size, output_size);

	biases->zero();
}

Tensor* Linear::forward(Tensor& input)
{
	this->input = &input;

	//dim3 blockDim(32, 32);
	//dim3 gridDim(ceil(((float)this->input->cols) / blockDim.x),
	//			 ceil(((float)weights->rows) / blockDim.y));

	dim3 dimGrid((weights->cols - 1) / TILE_WIDTH + 1,
				 (input.rows - 1) / TILE_WIDTH + 1, 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

	//forwardKernel << <gridDim, blockDim >> > (this->input->dev, weights->dev, output->dev, biases->dev, this->input->rows, this->input->cols, weights->rows,
	forwardKernel << <dimGrid, dimBlock >> > (this->input->dev, weights->dev, output->dev, biases->dev, this->input->rows, this->input->cols, weights->rows,
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
