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

	void fit(std::vector<Tensor> x, std::vector<Tensor> y, float lr, int epochs);
};