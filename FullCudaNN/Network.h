#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>

#include "Tensor.h"
#include "Layer.h"

class Network {
	public:

	void addLayer(Layer* layer);

	Tensor* forwardPass(Tensor& input);
	void backwardPass(Tensor& input, float lr);

	void fit(std::vector<Tensor*>& x_train, std::vector<Tensor*>& y_train, float lr, int epochs);

	float evaluate(std::vector<Tensor*>& x_test, std::vector<Tensor*>& y_test);

	private:
	std::vector<Layer*> layers;
};