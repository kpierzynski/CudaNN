#pragma once

#include <cstdio>
#include <iostream>
#include <vector>

class Tensor {
private:
	float* data;

	const int rows;
	const int cols;

public:
	Tensor(int rows, int cols);
	Tensor(const Tensor& other);
	~Tensor();

	void set_from( const std::vector<float>& data);

	void print();	// print tensor
	int size();		// returns rows * cols

	float get(int row, int col) const;
	void set(int row, int col, float value);

	Tensor& operator-=(float s);
	Tensor& operator+=(float s);
	Tensor& operator*=(float s);
	Tensor& operator/=(float s);

	Tensor operator*(float s) const;
	Tensor operator+(float s) const;
	Tensor operator-(float s) const;
	Tensor operator/(float s) const;

	Tensor operator*(const Tensor& t) const;
	Tensor operator-(const Tensor& t) const;
	Tensor operator+(const Tensor& t) const;
};