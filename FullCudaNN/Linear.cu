#include "Linear.h"

#include <cstdio>
#include <iostream>
#include <chrono>

#define TILED

#ifdef TILED
#define TILE_WIDTH 16
#endif

__global__ void forwardKernel(float* A, float* B, float* C, float* b, int rowsA, int colsA, int rowsB, int colsB) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rowsA && col < colsB) {
		float sum = 0;
		for (int ii = 0; ii < colsA; ii++) {
			sum += A[row * colsA + ii] * B[ii * colsB + col];
		}
		C[row * colsB + col] = sum + b[row * colsB + col];
	}
}

#ifdef TILED
// Compute C = A * B
__global__ void tiledForwardKernel(float* A, float* B, float* C, float* b, int rowsA, int colsA, int rowsB, int colsB) {
	__shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y,
		row = by * TILE_WIDTH + ty, col = bx * TILE_WIDTH + tx;
	float value = 0;

	for (int m = 0; m < (colsA - 1) / TILE_WIDTH + 1; ++m) {
		if (row < rowsA && m * TILE_WIDTH + tx < colsA)
			ds_M[ty][tx] = A[row * colsA + m * TILE_WIDTH + tx];
		else
			ds_M[ty][tx] = 0;
		if (col < colsB && m * TILE_WIDTH + ty < rowsB)
			ds_N[ty][tx] = B[(m * TILE_WIDTH + ty) * colsB + col];
		else
			ds_N[ty][tx] = 0;

		__syncthreads();
		for (int k = 0; k < TILE_WIDTH; ++k)
			value += ds_M[ty][k] * ds_N[k][tx];
		__syncthreads();
	}
	if (row < rowsA && col < colsB)
		C[row * colsB + col] = value + b[row * colsB + col];
}
#endif

#ifdef TILED
// Compute C = A * B^T
__global__ void tiledBackwardKernel(float* A, float* B, float* C, int rowsA, int colsA, int colsB) {
	__shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y,
		row = by * TILE_WIDTH + ty, col = bx * TILE_WIDTH + tx;
	float Pvalue = 0;

	for (int m = 0; m < (colsA - 1) / TILE_WIDTH + 1; ++m) {
		if (row < rowsA && m * TILE_WIDTH + tx < colsA)
			ds_M[ty][tx] = A[row * colsA + m * TILE_WIDTH + tx];
		else
			ds_M[ty][tx] = 0;
		if (col < colsB && m * TILE_WIDTH + ty < colsA)
			ds_N[tx][ty] = B[col * colsA + m * TILE_WIDTH + ty];
		else
			ds_N[tx][ty] = 0;

		__syncthreads();
		for (int k = 0; k < TILE_WIDTH; ++k)
			Pvalue += ds_M[ty][k] * ds_N[tx][k];
		__syncthreads();
	}
	if (row < rowsA && col < colsB)
		C[row * colsB + col] = Pvalue;
}
#endif


__global__ void backwardKernel(float* A, float* B, float* C, int rowsA, int colsA, int colsB)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < rowsA && col < colsB)
	{
		float sum = 0.0f;

		for (int i = 0; i < colsA; ++i)
			sum += A[row * colsA + i] * B[col * colsB + i];

		C[row * colsB + col] = sum;
	}
}


__global__ void updateWeightsKernel(float* W, const float* A, const float* B, float lr, int rowsW, int colsW, int rowsA, int colsA, int rowsB, int colsB) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < colsA && col < colsB) {
		float value = 0.0f;

		for (int i = 0; i < rowsB; i++)
			value += A[i * colsA + row] * B[i * colsB + col];

		W[row * colsW + col] -= lr * value;
	}
}

#ifdef TILED
// Compute C = A' * B
__global__ void tiledUpdateWeightsKernel(float* W, const float* A, const float* B, float lr, int rowsW, int colsW, int rowsA, int colsA, int rowsB, int colsB) {

	__shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y,
		row = by * TILE_WIDTH + ty, col = bx * TILE_WIDTH + tx;
	float Pvalue = 0;

	for (int m = 0; m < (rowsA - 1) / TILE_WIDTH + 1; ++m) {
		if (m * TILE_WIDTH + ty < rowsA && col < colsA)
			ds_M[ty][tx] = A[(m * TILE_WIDTH + ty) * colsA + col];
		else
			ds_M[ty][tx] = 0;
		if (row < rowsB && m * TILE_WIDTH + tx < colsB)
			ds_N[ty][tx] = B[row * colsB + m * TILE_WIDTH + tx];
		else
			ds_N[ty][tx] = 0;

		__syncthreads();
		for (int k = 0; k < TILE_WIDTH; ++k)
			Pvalue += ds_M[k][ty] * ds_N[k][tx];
		__syncthreads();
	}
	if (row < rowsB && col < colsA)
		W[row * colsA + col] -= Pvalue * lr;
}
#endif

__global__ void updateBiasKernel(float* A, float* b, int rowsA, int colsA, int rowsB, float lr) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < rowsA && col < colsA)
		b[row * colsA + col] -= lr * A[row * colsA + col];
}

Linear::Linear(int input_size, int output_size, int batch_size) : Layer(input_size, output_size)
{
	weights = new Tensor(input_size, output_size);
	biases = new Tensor(batch_size, output_size);

	gradient = new Tensor(batch_size, input_size);
	output = new Tensor(batch_size, output_size);

	biases->zero();
}

Tensor* Linear::forward(Tensor& input)
{
	this->input = &input;

	#ifdef TILED

	dim3 dimGrid((weights->cols - 1) / TILE_WIDTH + 1, (this->input->rows - 1) / TILE_WIDTH + 1, 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	tiledForwardKernel << <dimGrid, dimBlock >> > (this->input->dev, weights->dev, output->dev, biases->dev, this->input->rows, this->input->cols, weights->rows, weights->cols);

	#else

	dim3 dimBlock(32, 32);
	dim3 dimGrid(ceil(((float)this->input->cols) / dimBlock.x), ceil(((float)weights->rows) / dimBlock.y));
	forwardKernel << <dimGrid, dimBlock >> > (this->input->dev, weights->dev, output->dev, biases->dev, this->input->rows, this->input->cols, weights->rows, weights->cols);

	#endif

	return output;
}

Tensor* Linear::backward(Tensor& input, float lr)
{
	#ifdef TILED
	{
		dim3 dimGrid((weights->cols - 1) / TILE_WIDTH + 1, (input.rows - 1) / TILE_WIDTH + 1, 1);
		dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
		tiledBackwardKernel << <dimGrid, dimBlock >> > (input.dev, weights->dev, gradient->dev, input.rows, input.cols, weights->rows);
	}
	#else
	{
		dim3 dimBlock(32, 32);
		dim3 dimGrid((weights->cols + dimBlock.x - 1) / dimBlock.x, (input.rows + dimBlock.y - 1) / dimBlock.y);
		backwardKernel << <dimGrid, dimBlock >> > (input.dev, weights->dev, gradient->dev, input.rows, input.cols, weights->rows);
	}
	#endif

	{
		dim3 dimBlock(32, 32);
		dim3 dimGrid((weights->cols + dimBlock.x - 1) / dimBlock.x, (weights->rows + dimBlock.y - 1) / dimBlock.y);
		updateWeightsKernel << <dimGrid, dimBlock >> > (weights->dev, this->input->dev, input.dev, lr, weights->rows, weights->cols, this->input->rows, this->input->cols, input.rows, input.cols);
	}

	{
		dim3 dimBlock(32, 32);
		dim3 dimGrid((input.cols * input.rows + dimBlock.x) / dimBlock.x);
		updateBiasKernel << <dimGrid, dimBlock >> > (input.dev, biases->dev, input.rows, input.cols, biases->rows, lr);
	}

	return gradient;
}
