#include "Tanh.h"

Tanh::Tanh(int size) : Layer(size, size), input(Tensor(1,size)) {
}

Tensor Tanh::forward(Tensor& input) {
    this->input = input;
    Tensor output = input;

    for (int i = 0; i < output.cols; i++) {
        output.set(0, i, tanh(input.get(0, i)) );
    }

    return output;
}

Tensor Tanh::backward(Tensor& gradient, float lr) {
    Tensor output = input;

    for (int i = 0; i < output.cols; i++) {
        output.set(0, i, tanh(input.get(0, i)));
    }

    output = output * output * -1.0f + 1;

    return output * gradient;
}
