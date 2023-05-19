#pragma once
#include <cstdint>
#include <random>

class Layer {
public:
	int input_size;
	int output_size;

	Layer( int input_size, int output_size ) : input_size(input_size), output_size(output_size) {
		// inicjalizacja wag losowymi wartosciami
	}

	virtual uint8_t * forward(uint8_t * data) = 0;
	virtual uint8_t * backward(uint8_t * gradient) = 0;

protected:
	void generate_random_weights(float* weights, int size) {
		std::random_device dev;
		std::mt19937 gen(dev());

		std::uniform_real_distribution<float> unif(-0.5, 0.5);

		for (int i = 0; i < size; i++) {
			weights[i] = unif(gen);
		}
	}

};