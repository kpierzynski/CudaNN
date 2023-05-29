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

class LossMSE {
	public:
	static float calculate(Tensor y_pred, Tensor y_real) {
		Tensor diff = y_pred - y_real;
		Tensor loss = diff * diff;

		float error = loss.mean();

		return error;
	}

	static Tensor derivative(Tensor y_pred, Tensor y_real) {
		Tensor derivative = ((y_pred - y_real) * 2.0f) / y_pred.cols;
		return derivative;
	}
};

void Network::fit(std::vector<Tensor>& x_train, std::vector<Tensor>& y_train, float lr, int epochs)
{
	if (x_train.size() != y_train.size()) {
		throw std::invalid_argument("Wrong set sizes. Cannot perform fit.");
	}

	for (int epoch = 0; epoch < epochs; epoch++) {
		float loss = 0.0f;

		for (int i = 0; i < x_train.size(); i++) {
			Tensor output = forwardPass(x_train[i]);

			loss = LossMSE::calculate(output, y_train[i]);
			Tensor lossDerivative = LossMSE::derivative(output, y_train[i]);
			backwardPass(lossDerivative, lr);

			printf("step: %d, loss: %f                                               \r", i, loss);
		}

		std::cout << "Epoch: " << epoch << " Loss: " << loss << "                 " << std::endl;
	}
}


Tensor convert(Tensor& t) {
	Tensor result(t.rows, t.cols);
	for (int r = 0; r < t.rows; r++) {
		int pos = 0;

		for (int i = 0; i < t.cols; i++) {
			if (t.get(r, i) > t.get(r, pos))
				pos = i;
		}

		for (int i = 0; i < t.cols; i++) {
			result.set(r, i, 0.0f);

			if (i == pos) result.set(r, i, 1.0f);
		}
	}
	return result;
}

float Network::evaluate(std::vector<Tensor>& x_test, std::vector<Tensor>& y_test)
{
	if (x_test.size() != y_test.size()) {
		throw std::invalid_argument("Wrong set sizes. Cannot perform evaluation.");
	}

	int correct_predictions = 0;
	int total_predictions = 0;

	for (int i = 0; i < x_test.size(); i++) {
		Tensor output = forwardPass(x_test[i]);
		Tensor result = convert(output);

		for (int k = 0; k < y_test[i].rows; k++) {
			if (result[k] == y_test[i][k]) correct_predictions++;
			total_predictions++;
		}

	}

	float accuracy = ((float)correct_predictions / total_predictions) * 100.0f;

	std::cout << "Evaluation Results: " << std::endl;
	std::cout << "Total Samples: " << x_test.size() * x_test[0].rows << std::endl;
	std::cout << "Correct predictions: " << correct_predictions << std::endl;
	std::cout << "Accuracy: " << accuracy << "%" << std::endl;

	return accuracy;
}


Tensor Network::predict(Tensor input)
{
	return forwardPass(input);
}


