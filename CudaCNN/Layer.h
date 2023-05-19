#pragma once
#include <cstdint>
#include <random>

class Layer {
public:
	int input_size;
	int output_size;

	Layer(int input_size, int output_size);

	virtual uint8_t * forward(uint8_t * data) = 0;
	virtual uint8_t * backward(uint8_t * gradient) = 0;

protected:
	void generate_random_weights(float* weights, int size);
};