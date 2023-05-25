#pragma once

#include <cstdio>
#include <iostream>
#include <vector>
#include <random>

class Tensor {
	public:
	//private:
	float* data;

	int rows;
	int cols;

	public:
	Tensor(int rows, int cols);
	Tensor(const Tensor& other);
	~Tensor();

	void set_from(const std::vector<float>& data);
	void set_from(const float* data, int size);
	void set_random();

	void print();		// print tensor
	int size() const;	// returns rows * cols

	float get(int row, int col) const;
	void set(int row, int col, float value);

	float mean() const;
	Tensor transpose() const;

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

	Tensor& operator+=(const Tensor& t);
	Tensor& operator-=(const Tensor& t);

	Tensor& operator=(const Tensor& t);
	bool operator==(const Tensor& t);
	bool operator!=(const Tensor& t);

	friend std::ostream& operator<< (std::ostream& stream, const Tensor& t);
};