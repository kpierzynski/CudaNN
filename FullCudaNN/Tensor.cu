#include "Tensor.h"
#include <iostream>
#include <random>

float generateRandomNumber() {
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

	return dist(gen);
}

Tensor::Tensor(int rows, int cols) : rows(rows), cols(cols)
{
	this->host = new float[rows * cols];
	cudaError_t cudaStatus = cudaMalloc(&this->dev, rows * cols * sizeof(float));

	for (int i = 0; i < rows * cols; i++) {
		//this->host[i] = (i & 1) ? -0.111f : 0.111f;
		this->host[i] = generateRandomNumber();
	}
	this->host2dev();
}

Tensor::Tensor(int rows, int cols, float * data) : rows(rows), cols(cols)
{
	this->host = new float[rows * cols];
	for (int i = 0; i < rows * cols; i++) {
		this->host[i] = data[i];
	}

	cudaError_t cudaStatus = cudaMalloc(&this->dev, rows * cols * sizeof(float));

	this->host2dev();
}

__global__  void copyKernel(float * dest, float * src, size_t size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < size)
	{
		dest[i] = src[i];
	}
}

Tensor::Tensor(const Tensor& other) : rows(other.rows), cols(other.cols)
{
	this->host = new float[other.rows * other.cols];
	for (int i = 0; i < other.rows * other.cols; i++) {
		this->host[i] = other.host[i];
	}

	cudaError_t cudaStatus = cudaMalloc(&this->dev, other.rows * other.cols * sizeof(float));

	int blockSize = 256;
	int numBlocks = (other.rows * other.cols + blockSize - 1) / blockSize;
	copyKernel << <numBlocks, blockSize >> > (this->dev, other.dev, other.rows * other.cols);

}

Tensor::~Tensor()
{
	delete[] this->host;
	cudaFree(this->dev);
}

void Tensor::dev2host()
{
	cudaError_t cudaStatus = cudaMemcpy(this->host, this->dev, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
}

void Tensor::host2dev()
{
	cudaError_t cudaStatus = cudaMemcpy(this->dev, this->host, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
}

void Tensor::zero() {
	std::memset(this->host, 0, b_size());
	this->host2dev();
}

void Tensor::print() {
	this->dev2host();

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			std::cout << host[cols * r + c] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

int Tensor::b_size()
{
	return rows * cols * sizeof(float);
}

float& Tensor::operator[](const int index) {
	return host[index];
}