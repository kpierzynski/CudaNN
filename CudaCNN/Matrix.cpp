#include "Matrix.h"

Matrix::Matrix(int rows, int columns) : rows(rows), columns(columns), data(rows* columns) {
    std::random_device dev;
    std::mt19937 gen(dev());

    std::uniform_real_distribution<float> unif(-0.5, 0.5);

    for (int i = 0; i < rows * columns; i++) {
        data[i] = unif(gen);
    }
}

int Matrix::getRows() const {
    return rows;
}

int Matrix::getColumns() const {
    return columns;
}

int Matrix::getElement(int row, int column) const {
    return data[row * columns + column];
}

void Matrix::setElement(int row, int column, int value) {
    data[row * columns + column] = value;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (columns != other.rows) {
        throw std::runtime_error("Incompatible matrix dimensions for multiplication");
    }

    Matrix result(rows, other.columns);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.columns; j++) {
            int sum = 0;
            for (int k = 0; k < columns; k++) {
                sum += getElement(i, k) * other.getElement(k, j);
            }
            result.setElement(i, j, sum);
        }
    }

    return result;
}