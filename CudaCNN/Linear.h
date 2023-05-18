#pragma once

#include "Layer.h"


class Linear : Layer {
public:
	Linear(int input_size, int output_size) : Layer(input_size, output_size) {

	}

	void forward(uint8_t * data) override {

	}

	void backward(uint8_t * gradient) override {

	}
};