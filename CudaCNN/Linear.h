#pragma once

#include "Layer.h"
#include "Matrix.h"

class Linear : public Layer {
public:
	Linear(int input_size, int output_size);

	std::vector<float> forward(std::vector<float>& data) override;
	std::vector<float> backward(std::vector<float>& gradient, float lr) override;
private:
	Matrix * weights;
	Matrix * bias;

	Matrix* input;
};