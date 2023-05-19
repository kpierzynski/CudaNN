#pragma once

#include <vector>

#include "Layer.h"

class Network {
public:
	std::vector<Layer> layers;

	Network();

	void addLayer(Layer& layer);
	void fit(float lr = 0.01f, int epochs = 4);
	void predict();

};