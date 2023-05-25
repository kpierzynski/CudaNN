#pragma once

#include <cmath>

#include "Layer.h"

class Tanh : public Layer {
    public:
    Tanh(int size);
    Tensor forward(Tensor& input) override;
    Tensor backward(Tensor& gradient, float lr) override;

    private:
    Tensor input;
};
