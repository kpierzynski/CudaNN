#include "Matrix.h"

Matrix::Matrix(int rows, int columns) : rows(rows), columns(columns), data(rows* columns) {
    std::random_device dev;
    std::mt19937 gen(dev());

    std::uniform_real_distribution<float> unif(-0.5, 0.5);

    for (int i = 0; i < rows * columns; i++) {
        data[i] = unif(gen);
    }
}

Matrix::Matrix(int rows, int columns, std::vector<float> values) : rows(rows), columns(columns) {
    if (values.size() != rows * columns ) {
        throw std::invalid_argument("Wrong size of given vector of values.");
    }

    for (float value : values) {
        data.push_back(value);
    }
}

int Matrix::getRows() const {
    return rows;
}

int Matrix::getColumns() const {
    return columns;
}

float Matrix::getElement(int row, int column) const {
    return data[row * columns + column];
}

void Matrix::setElement(int row, int column, float value) {
    data[row * columns + column] = value;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (columns != other.rows) {
        throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
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


Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || columns != other.columns) {
        throw std::invalid_argument("Incompatible matrix dimensions for addition");
    }

    Matrix result(rows, columns);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            result.setElement(i, j, getElement(i, j) + other.getElement(i, j));
        }
    }

    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(columns, rows);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            result.setElement(j, i, getElement(i, j));
        }
    }

    return result;
}

Matrix Matrix::operator*(float scalar) const {
    Matrix result(rows, columns);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            result.setElement(i, j, static_cast<int>(getElement(i, j) * scalar));
        }
    }

    return result;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    if (rows != other.rows || columns != other.columns) {
        throw std::invalid_argument("Incompatible matrix dimensions for subtraction");
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            setElement(i, j, getElement(i, j) - other.getElement(i, j));
        }
    }

    return *this;
}

Matrix Matrix::operator-(float scalar) const {
    Matrix result(rows, columns);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            result.setElement(i, j, getElement(i, j) - static_cast<int>(scalar));
        }
    }

    return result;
}

Matrix Matrix::operator/(float scalar) const {
    if (scalar == 0.0f) {
        // Handle division by zero as per your requirements
        throw std::invalid_argument("Division by zero");
    }

    Matrix result(rows, columns);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            result.setElement(i, j, static_cast<int>(getElement(i, j) / scalar));
        }
    }

    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || columns != other.columns) {
        throw std::invalid_argument("Incompatible matrix dimensions for subtraction");
    }

    Matrix result(rows, columns);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            result.setElement(i, j, getElement(i, j) - other.getElement(i, j));
        }
    }

    return result;
}