#pragma once

#include <iostream>

#include "Layer.h"
#include "Matrix.h"

class ReLU : public Layer {
public:
	ReLU(int size);

	Matrix& forward(Matrix& data) override;
	Matrix& backward(Matrix& gradient, float lr) override;

	Matrix input;
};