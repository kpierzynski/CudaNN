#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Layer.h"

class Tanh : public Layer {
	private:
	int batch_size;

	public:
	Tanh(int size, int batch_size);

	Tensor* forward(Tensor& input) override;
	Tensor* backward(Tensor& input, float lr) override;

	Tensor* input;
	Tensor* output;

	Tensor* dA;
};