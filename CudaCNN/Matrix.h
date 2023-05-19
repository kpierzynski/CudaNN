#pragma once

#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
#include <random>

class Matrix {
public:
    Matrix(int rows, int columns);

    int getRows() const;
    int getColumns() const;

    int getElement(int row, int column) const;
    void setElement(int row, int column, int value);

    Matrix operator*(const Matrix& other) const;

private:
    int rows;
    int columns;
    std::vector<float> data;
};
