#include "Tensor.h"
#include <iostream>
#include <random>

double generateRandomNumber() {
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dist(-1.0, 1.0);

	return dist(gen);
}

Tensor::Tensor(int rows, int cols) : rows(rows), cols(cols)
{
	this->host = new float[rows * cols];
	cudaMalloc(&this->dev, rows * cols * sizeof(float));
	for (int i = 0; i < rows * cols; i++) {
		this->host[i] = (i&1) ? 0.569f : -0.569f;
	}
	this->host2dev();
}

Tensor::Tensor(int rows, int cols, float * data) : rows(rows), cols(cols)
{
	this->host = new float[rows * cols];
	std::memcpy(this->host, data, rows * cols * sizeof(float));
	cudaMalloc(&this->dev, rows * cols * sizeof(float));
	this->host2dev();
}

Tensor::Tensor(const Tensor& other) : rows(other.rows), cols(other.cols)
{
	this->host = new float[rows * cols];
	cudaMalloc(&this->dev, rows * cols * sizeof(float));

	std::memcpy(this->host, other.host, rows * cols * sizeof(float));
	this->host2dev();
}

Tensor::~Tensor()
{
	//std::cout << "Tensor destroyed: " << rows << " " << cols << std::endl;
	cudaFree(this->dev);
	delete[] this->host;
}

void Tensor::dev2host()
{
	cudaMemcpy(this->host, this->dev, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
}

void Tensor::host2dev()
{
	cudaMemcpy(this->dev, this->host, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
}

void Tensor::zero() {
	std::memset(this->host, 0, b_size());
	this->host2dev();
}

void Tensor::print() {
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