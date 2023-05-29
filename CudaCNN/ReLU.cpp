#include "ReLU.h"

ReLU::ReLU(int size) : Layer(size, size), input(Tensor(1, size)) {
}

Tensor ReLU::forward(Tensor& input) {
    this->input = input;
    Tensor output = input;

    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            output.set(i, j, std::max(0.0f, input.get(i, j)));
        }
    }

    return output;
}

Tensor ReLU::backward(Tensor& gradient, float lr) {
    Tensor output = input;

    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            output.set(i, j, input.get(i, j) <= 0.0f ? 0.0f : gradient.get(i,j) );
        }
    }

    return output;
}
