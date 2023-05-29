#include "Tanh.h"

Tanh::Tanh(int size) : Layer(size, size), input(Tensor(1,size)) {
}

Tensor Tanh::forward(Tensor& input) {
    this->input = input;
    Tensor output = input;

    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            output.set(i, j, tanh(input.get(i, j)));
        }
    }

    return output;
}

Tensor Tanh::backward(Tensor& gradient, float lr) {
    Tensor output = input;

    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            output.set(i, j, tanh(input.get(i, j)));
        }
    }

    output = output * output * -1.0f + 1;

    return output * gradient;
}
