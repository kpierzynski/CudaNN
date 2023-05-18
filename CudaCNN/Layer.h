#pragma once
#include <cstdint>

class Layer {
public:
	int input_size;
	int output_size;

	Layer( int input_size, int output_size ) : input_size(input_size), output_size(output_size) {
		// inicjalizacja wag losowymi wartosciami
	}

	virtual void forward(uint8_t * data) = 0;
	virtual void backward(uint8_t * gradient) = 0;

};