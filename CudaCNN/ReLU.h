#pragma once

#include "Layer.h"

#define RELU_MAX(a) (((a)>0) ? ((a)) : (0))

class ReLU : public Layer {
    public:
    ReLU(int size);
    Tensor forward(Tensor& input) override;
    Tensor backward(Tensor& gradient, float lr) override;

    private:
    Tensor input;
};
