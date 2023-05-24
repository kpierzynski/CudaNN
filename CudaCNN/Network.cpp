#include "Network.h"

void Network::addLayer(Layer* layer)
{
	layers.push_back(layer);
}

Tensor Network::forwardPass(Tensor input)
{
	for (Layer* layer : layers) {
		input = layer->forward(input);
	}

	return input;
}

void Network::backwardPass(Tensor input, float lr)
{
	for (int i = layers.size() - 1; i >= 0; --i) {
		input = layers[i]->backward(input, lr);
	}
}

class Loss {
	public:
	static float calculate(Tensor y_pred, Tensor y_read) {
		Tensor diff = y_pred - y_read;
		Tensor loss = diff * diff;

		float error = loss.mean();

		return error;
	}

	static Tensor derivative(Tensor y_pred, Tensor y_real) {
		Tensor derivative = ((y_pred - y_real) * 2.0f) / y_pred.size();
		return derivative;
	}
};

void Network::fit(std::vector<Tensor> x_set, std::vector<Tensor> y_set, float lr, int epochs)
{
	if (x_set.size() != y_set.size()) {
		throw std::invalid_argument("Wrong set sizes. Cannot perform fit.");
	}

	for (int epoch = 0; epoch < epochs; epoch++) {
		float loss = 0.0f;

		for (int i = 0; i < x_set.size(); i++) {
			Tensor output = forwardPass(x_set[i]);
			loss = Loss::calculate(output, y_set[i]);

			Tensor lossDerivative = Loss::derivative(output, y_set[i]);
			backwardPass(lossDerivative, lr);

			printf("step: %d, loss: %f                                    \r", i, loss);
		}

		std::cout << "Epoch: " << epoch << " Loss: " << loss << std::endl;
	}
}
