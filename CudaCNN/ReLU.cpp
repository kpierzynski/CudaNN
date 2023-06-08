#include "ReLU.h"

ReLU::ReLU(int size) : Layer(size, size), input(Tensor(1, size)) {
}

float relu(float x) {
    return (x > 0) ? x : 0;
}

float reluDerivative(float x) {
    return (x > 0) ? 1 : 0;
}

Tensor ReLU::forward(Tensor& input) {
    this->input = input;
    Tensor output = input;


    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            output.set(i, j, relu(input.get(i, j)));
        }
    }

    return output;
}

Tensor ReLU::backward(Tensor& gradient, float lr) {
    Tensor output = input;

    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            output.set(i, j, reluDerivative(input.get(i, j)));
        }
    }

    return output * gradient;
}
