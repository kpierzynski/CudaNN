#pragma once

#include "Layer.h"
#include "Matrix.h"

class Linear : public Layer {
public:
	Linear(int input_size, int output_size);

	Matrix& forward(Matrix& data) override;
	Matrix& backward(Matrix& gradient, float lr) override;
private:
	Matrix * weights;
	Matrix * bias;

	Matrix input;
};