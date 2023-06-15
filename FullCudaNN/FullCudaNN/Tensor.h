#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstring>

class Tensor {
	public:

	Tensor(int rows, int cols);
	Tensor(int rows, int cols, float* data);
	Tensor(const Tensor& other);

	~Tensor();

	void dev2host();
	void host2dev();

	void zero();

	void print();
	int b_size();	// returns size in bytes!

	float& operator[](const int index);

	int rows;
	int cols;

	float* host;
	float* dev;
};