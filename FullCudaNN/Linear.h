#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Layer.h"

class Linear : public Layer {
	public:
	Linear(int input_size, int output_size, int batch_size);

	Tensor * forward(Tensor& input) override;
	Tensor * backward(Tensor& input, float lr) override;

	private:
	Tensor* weights;
	Tensor* biases;
	Tensor* input;
	Tensor* output;

	Tensor* gradient;
};