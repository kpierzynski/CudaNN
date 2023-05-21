#include "ReLU.h"

ReLU::ReLU(int size) : Layer(size, size), input(1,1) {
}

Matrix& ReLU::forward(Matrix& input) {
    this->input = input;

    //std::cout << "Relu" << input.getColumns() << " " << input.getRows() << std::endl;

    Matrix* result = new Matrix(1, input_size);

    for (int i = 0; i < input_size; i++) {
        result->setElement(0, i, std::max(0.0f, input.getElement(0,i)) );
    }

    return *result;
}

Matrix& ReLU::backward(Matrix& input, float lr) {

    Matrix* result = new Matrix(1, input_size);

    for (int i = 0; i < output_size; i++) {
        result->setElement(0, i, input.getElement(0,i) * (this->input.getElement(0,i) > 0 ? 1 : 0) );
    }

    return *result;
}