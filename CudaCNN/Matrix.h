#pragma once

#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
#include <random>

class Matrix {
public:
    std::vector<float> data;

    Matrix(int rows, int columns);
    Matrix(int rows, int columns, std::vector<float> values);

    int getRows() const;
    int getColumns() const;

    float getElement(int row, int column) const;
    void setElement(int row, int column, float value);

    Matrix transpose() const;

    Matrix operator*(const Matrix& other) const;
    Matrix operator*(float scalar) const;
    Matrix operator-(float scalar) const;
    Matrix operator/(float scalar) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator+(const Matrix& other) const;
    Matrix& Matrix::operator-=(const Matrix& other);

private:
    int rows;
    int columns;
};
