#include "Network.h"

Network::Network() {

}

void Network::addLayer(Layer& layer) {
	this->layers.push_back(layer);
}

void Network::fit(std::vector<uint8_t*> x_input, std::vector<int> y_input, float lr, int epochs) {
	for (int epoch = 0; epoch < epochs; epoch++) {
		// for each item in x_input train set
		for (uint8_t* sample : x_input) {

			//FORWARD STEP

			uint8_t* data = sample;
			// go through all layers and compute forward result
			for (Layer& layer : layers) {
				data = layer.forward(data);
			}
			float error = loss_function(sample, data);

			//BACKWARD STEP
			float back_error = loss_deriv_function(sample, data);

			// Loop over layers, but from end.
			for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
				Layer& layer = *it;

				back_error = layer.backward(back_error, lr);

			}
		}
	}
}

void Network::predict() {

}