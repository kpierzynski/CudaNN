#pragma once

#include "Layer.h"


class Linear : Layer {
public:
	Linear(int input_size, int output_size);

	uint8_t * forward(uint8_t* data) override;
	uint8_t * backward(uint8_t* gradient) override;
};