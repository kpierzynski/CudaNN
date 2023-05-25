#pragma once

#include <iostream>
#include <vector>

#include "Layer.h"
#include "Tensor.h"

class Network {
	private:
	std::vector<Layer*> layers;

	public:
	void addLayer(Layer* layer);

	Tensor forwardPass(Tensor input);
	void backwardPass(Tensor input, float lr);

	void fit(std::vector<Tensor>& x_train, std::vector<Tensor>& y_train, float lr, int epochs);
	float evaluate(std::vector<Tensor>& x_test, std::vector<Tensor>& y_test);
	Tensor predict(Tensor input);
};