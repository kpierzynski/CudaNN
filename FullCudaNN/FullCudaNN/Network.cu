#include "Network.h"
#include <iostream>
#include "MSE.h"

void Network::addLayer(Layer* layer)
{
	this->layers.push_back(layer);
}

Tensor& Network::forwardPass(Tensor& input)
{
	for (Layer* layer : this->layers) {
		input = layer->forward(input);
	}
	
	cudaDeviceSynchronize();

	return input;
}

void Network::backwardPass(Tensor& input, float lr)
{
	for (int i = layers.size() - 1; i >= 0; --i) {
		input = layers[i]->backward(input, lr);
	}

	cudaDeviceSynchronize();
}

void Network::fit(std::vector<Tensor*>& x_train, std::vector<Tensor*>& y_train, float lr, int epochs)
{
	std::cout << "Network::fit" << std::endl;

	if (x_train.size() != y_train.size()) {
		throw std::invalid_argument("Wrong set sizes. Cannot perform fit.");
	}

	Tensor output(x_train[0]->rows, y_train[0]->cols);
	for (int epoch = 0; epoch < epochs; epoch++) {
		float loss = 0.0f;

		for (int i = 0; i < x_train.size(); i++) {
			output = forwardPass(*x_train[i]);
			loss = MSE::cost(output, *y_train[i]);
			//output.dev2host();
			//output.print();
			//y_train[i]->dev2host();
			//y_train[i]->print();
			Tensor* lossDerivative = MSE::derivative(output, *y_train[i]);
			
			//lossDerivative->dev2host();
			//lossDerivative->print();

			backwardPass(*lossDerivative, lr);

			//delete lossDerivative;	// is this here required?
			printf("Step: %d, loss: %f                                    \r", i, loss);
		}

		printf("Epoch: %d, Loss: %f                                    \r\n", epoch+1, loss);
	}
}


// do it private methods of network class
int compare(Tensor& a, Tensor& b) {
	int cnt = 0;

	for (int r = 0; r < a.rows; r++) {
		for (int c = 0; c < a.cols; c++) {
			if (a[r * a.cols + c] == 1.0f) {
				if (b[r * a.cols + c] == 1.0f) cnt++;
				else break;
			}
		}
	}

	return cnt;
}

void convert(Tensor& t) {
	for (int r = 0; r < t.rows; r++) {
		int pos = -1;
		float max = 0.0f;

		for (int c = 0; c < t.cols; c++) {
			if (t[r * t.cols + c] > max) {
				max = t[r * t.cols + c];
				pos = c;
			}
		}

		for (int c = 0; c < t.cols; c++) {
			if (c == pos) t[r * t.cols + c] = 1.0f;
			else t[r * t.cols + c] = 0.0f;
		}
	}
}
float Network::evaluate(std::vector<Tensor*>& x_test, std::vector<Tensor*>& y_test)
{
	if (x_test.size() != y_test.size()) {
		throw std::invalid_argument("Wrong set sizes. Cannot perform evaluation.");
	}

	int correct_predictions = 0;
	int total_predictions = 0;

	for (int i = 0; i < x_test.size(); i++) {
		Tensor output = forwardPass(*x_test[i]);
		convert(output);

		correct_predictions += compare(output, *y_test[i]);
		total_predictions += output.rows;

	}

	float accuracy = ((float)correct_predictions / total_predictions) * 100.0f;

	std::cout << "Evaluation Results: " << std::endl;
	std::cout << "Total Samples: " << x_test.size() * x_test[0]->rows << std::endl;
	std::cout << "Correct predictions: " << correct_predictions << std::endl;
	std::cout << "Accuracy: " << accuracy << "%" << std::endl;

	return accuracy;
}