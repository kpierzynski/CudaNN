#pragma once
#include <cstdint>

class LossFunction {
public:
	LossFunction() {

	}

	virtual float get_loss() = 0;
	virtual uint8_t* calculate() = 0;
};