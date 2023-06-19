#include "MSE.h"
#include <stdexcept>
#include <iostream>

__global__ void cudaMSELoss(const float* predictions, const float* targets, float* loss, int size) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < size) {
		float diff = predictions[idx] - targets[idx];
		atomicAdd(loss, diff * diff);
	}
}

__global__ void cudaMSELossDerivative(float* predictions, float* targets, float* derivatives, int size, int batch_size)
{
	//printf("cudaMSELossDerivative\r\n");

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

	float* loss_dev;
	float loss = -13.0f;
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc(&loss_dev, sizeof(float));
	cudaStatus = cudaMemset(loss_dev, 0, sizeof(float));

	int size = y_real.rows * y_real.cols;
	int blockSize = size;
	int gridSize = (size + blockSize) / blockSize;

	cudaMSELoss << <gridSize, blockSize >> > (y_pred.dev, y_real.dev, loss_dev, size);
	cudaStatus = cudaGetLastError();

	cudaMemcpy(&loss, loss_dev, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(loss_dev);

	return loss / y_real.cols / y_real.rows;
}

Tensor * MSE::derivative(Tensor& y_pred, Tensor& y_real)
{
	if (y_pred.b_size() != y_real.b_size()) {
		throw std::invalid_argument("Wrong set sizes. Cannot perform fit.");
	}

	int size = y_real.cols;
	int batch_size = y_real.rows;
	int blockSize = (size * batch_size);
	int gridSize = (size + blockSize) / blockSize;

	Tensor * deriv = new Tensor(y_real.rows, y_real.cols);

	cudaMSELossDerivative << <gridSize, blockSize >> > (y_pred.dev, y_real.dev, deriv->dev, size, batch_size*size);

	return deriv;
}
