#include "Network.h"
#include "Matrix.h"

#include <iostream>

Network::Network() {

}

float Network::loss_function(std::vector<float>& y_real, std::vector<float>& y_pred) {
	float sum = 0;

	if (y_real.size() != y_pred.size()) {
		std::cout << "y_real.size(): " << y_real.size() << " y_pred.size(): " << y_pred.size() << std::endl;
		throw std::invalid_argument("y sizes are different.");
	}

	int size = y_real.size();
	for (int i = 0; i < size; i++) {
		sum += (y_real[i] - y_pred[i]) * (y_real[i] - y_pred[i]);
	}

	return sum / size;
}

Matrix& Network::loss_deriv_function(std::vector<float>& y_real, std::vector<float>& y_pred) {
	if (y_real.size() != y_pred.size()) {
		std::cout << "y_real.size(): " << y_real.size() << " y_pred.size(): " << y_pred.size() << std::endl;
		throw std::invalid_argument("y sizes are different.");
	}

	int size = y_real.size();

	Matrix m_y_real(1, size, y_real);
	Matrix m_y_pred(1, size, y_pred);

	Matrix* result = new Matrix(1, size);
	*result = ((m_y_real - m_y_pred) * 2.0f / size);

	return *result;
}

void Network::addLayer(Layer * layer) {
	this->layers.push_back(layer);
}

void Network::fit(std::vector<std::vector<float>>& x_input, std::vector<int>& y_input, float lr, int epochs) {
	if (x_input.size() != y_input.size()) {
		throw std::runtime_error("Invalid input dataset sizes.");
	}
	int size = x_input.size();

	for (int epoch = 0; epoch < epochs; epoch++) {
		// for each item in x_input train set
		float error = 0;

		for (int i = 0; i < size; i++ ) {

			//FORWARD STEP

			std::vector<float> v_data = x_input[i];
			Matrix * data = new Matrix(1, x_input[i].size(), v_data);
			// go through all layers and compute forward result

			for (Layer * layer : layers) {
				*data = layer->forward(*data);
				//std::cout << "Inside forward layer loop" << data->getColumns() << " " << data->getRows() << std::endl;
			}
			error = loss_function(one_hot_encoder(y_input[i]), data->data);

			//BACKWARD STEP
			Matrix back_error = loss_deriv_function(one_hot_encoder(y_input[i]), data->data);

			// loop over layers, but from end.
			for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
				Layer * layer = *it;
			
				back_error = layer->backward(back_error, lr);
				//std::cout << "BACKWARD loop" << back_error.getColumns() << " " << back_error.getRows() << std::endl;

			}

			delete data;
		}

		std::cout << "Epoch done: " << epoch << " error: " << error/size << std::endl;
	}
}

std::vector<float> Network::one_hot_encoder(int label) {
	std::vector<float> vec;

	for (int i = 0; i < 10; i++) {
		if (i == label)
			vec.push_back(1.0f);
		else
			vec.push_back(0.0f);
	}

	return vec;
}

void Network::predict() {

}