#include "MSE.h"
#include <stdexcept>
#include <iostream>
#include <vector>

#define BLOCK_SIZE 512

__global__ void cudaMSELoss(const float* predictions, const float* targets, float* loss, int size) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < size) {
		float diff = predictions[idx] - targets[idx];
		atomicAdd(loss, diff * diff);
	}
}

__global__ void cudaMSELossReduction(const float* predictions, const float* targets, float* output, int len) {
	__shared__ float partial_sum[2 * BLOCK_SIZE];
	int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;

	if (start + t < len) {
		float value = predictions[start + t] - targets[start + t];
		partial_sum[t] = value * value;
	}
	else
		partial_sum[t] = 0;
	if (start + BLOCK_SIZE + t < len) {
		float value = predictions[start + BLOCK_SIZE + t] - targets[start + BLOCK_SIZE + t];
		partial_sum[BLOCK_SIZE + t] = value * value;
	}
	else
		partial_sum[BLOCK_SIZE + t] = 0;

	for (int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
		__syncthreads();
		if (t < stride)
			partial_sum[t] += partial_sum[t + stride];
	}

	if (t == 0)
		output[blockIdx.x] = partial_sum[0];
}

__global__ void cudaMSELossDerivative(float* predictions, float* targets, float* derivatives, int size, int batch_size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < batch_size)
	{
		derivatives[idx] = 2.0 * (predictions[idx] - targets[idx]) / size;
	}
}

float MSE::cost(Tensor& y_pred, Tensor& y_real)
{
	if (y_pred.b_size() != y_real.b_size()) {
		throw std::invalid_argument("Wrong set sizes. Cannot perform fit.");
	}

	int size = y_real.rows * y_real.cols;

	int input_size = size;
	int output_size = input_size / (BLOCK_SIZE << 1);
	if (input_size % (BLOCK_SIZE << 1)) {
		output_size++;
	}
	
	static float* loss_dev = nullptr;
	std::vector<float> loss_host(output_size);

	if (loss_dev == nullptr) {
		cudaMalloc(&loss_dev, output_size * sizeof(float));
	}
	cudaMemset(loss_dev, 0, output_size * sizeof(float));

	dim3 dimGrid(output_size, 1, 1);
	dim3 dimBlock(BLOCK_SIZE, 1, 1);

	cudaMSELossReduction<< <dimGrid, dimBlock >> > (y_pred.dev, y_real.dev, loss_dev, input_size);

	cudaMemcpy(loss_host.data(), loss_dev, output_size * sizeof(float), cudaMemcpyDeviceToHost);

	float loss = 0.0f;
	for (int i = 0; i < loss_host.size(); i++) {
		loss += loss_host[i];
	}

	return loss / y_real.cols / y_real.rows;
}

void MSE::derivative(Tensor& result, Tensor& y_pred, Tensor& y_real)
{
	if (y_pred.b_size() != y_real.b_size()) {
		throw std::invalid_argument("Wrong set sizes. Cannot perform fit.");
	}

	int size = y_real.cols;
	int batch_size = y_real.rows;

	{
		int dimBlock = (size * batch_size);
		int dimGrid = (size + dimBlock) / dimBlock;
		cudaMSELossDerivative << <dimGrid, dimBlock >> > (y_pred.dev, y_real.dev, result.dev, size, batch_size * size);
	}
}
