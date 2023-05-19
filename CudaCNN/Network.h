#pragma once

#include <vector>

#include "Layer.h"

class Network {
public:
	Network();

	void addLayer(Layer * layer);
	void fit(	std::vector<uint8_t*>& x_input,
				std::vector<int>& y_input,
				float lr = 0.01f,
				int epochs = 4
			);
	void predict();

private:
	std::vector<Layer*> layers;

};