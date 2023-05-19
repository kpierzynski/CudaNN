#pragma once

#include <vector>

#include "Layer.h"

class Network {
public:
	Network();

	void addLayer(Layer * layer);
	void fit(std::vector<std::vector<float>>& x_input, std::vector<int>& y_input, float lr, int epochs);
	void predict();

private:
	std::vector<Layer*> layers;

	float Network::loss_function(std::vector<float>& y_real, std::vector<float>& y_pred);
	std::vector<float> Network::loss_deriv_function(std::vector<float>& y_real, std::vector<float>& y_pred);

	std::vector<float> Network::one_hot_encoder(int label);

};