#include "Network.h"

Network::Network() {

}

void Network::addLayer(Layer& layer) {
	this->layers.push_back(layer);
}

void Network::fit(float lr, int epochs) {
	for (int epoch = 0; epoch < epochs; epoch++) {

		// note: std::reverse for reverse layers in vector for backpropagation
	}
}

void Network::predict() {

}