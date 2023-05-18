#pragma once

#include "Layer.h"

class ReLU : Layer {
public:
	ReLU(int size) : Layer(size, size) {

	}

	void forward(uint8_t* data) override {

	}

	void backward(uint8_t* gradient) override {

	}

};