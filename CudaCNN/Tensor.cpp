#include "Tensor.h"

Tensor::Tensor(int rows, int cols) : rows(rows), cols(cols) {
	data = new float[rows * cols];
}

Tensor::Tensor(const Tensor& other) : rows(other.rows), cols(other.cols) {
	data = new float[rows * cols];

	std::copy(other.data, other.data + (rows * cols), data);
}

Tensor::~Tensor()
{
	delete[] data;
}

void Tensor::set_from(const std::vector<float>& data)
{
	if (data.size() != rows * cols) {
		throw std::invalid_argument("Wrong size of vector.");
	}

	int i = 0;
	for (float value : data) {
		this->data[i++] = value;
	}
}

void Tensor::print() {

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			std::cout << data[r * cols + c] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

int Tensor::size()
{
	return rows * cols;
}

float Tensor::get(int row, int col) const
{
	return data[row * cols + col];
}

void Tensor::set(int row, int col, float value)
{
	data[row * cols + col] = value;
}

Tensor& Tensor::operator-=(float s)
{
	for (int i = 0; i < rows * cols; i++) {
		data[i] -= s;
	}

	return *this;
}

Tensor& Tensor::operator+=(float s)
{
	for (int i = 0; i < rows * cols; i++) {
		data[i] += s;
	}

	return *this;
}

Tensor& Tensor::operator*=(float s)
{
	for (int i = 0; i < rows * cols; i++) {
		data[i] *= s;
	}

	return *this;
}

Tensor& Tensor::operator/=(float s)
{
	if (s == 0.0f) {
		throw std::invalid_argument("Cannot divide by zero.");
	}

	for (int i = 0; i < rows * cols; i++) {
		data[i] /= s;
	}

	return *this;
}

Tensor Tensor::operator*(const Tensor& t) const
{
	if (cols != t.rows) {
		throw std::invalid_argument("Cannot perform matrix multiplcation. Wrong dimentions");
	}

	Tensor result(rows, t.cols);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < t.cols; j++) {
			float value = 0.0f;
			for (int k = 0; k < cols; k++) {
				value += get(i, k) * t.get(k, j);
			}
			result.set(i, j, value);
		}
	}

	return result;

}

Tensor Tensor::operator-(const Tensor& t) const
{
	if (rows != t.rows || cols != t.cols) {
		throw std::invalid_argument("Cannot perform subtraction. Wrong dimensions");
	}

	Tensor result(rows, cols);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			float value = get(i, j) - t.get(i, j);
			result.set(i, j, value);
		}
	}

	return result;
}

Tensor Tensor::operator+(const Tensor& t) const
{
	if (rows != t.rows || cols != t.cols) {
		throw std::invalid_argument("Cannot perform addition. Wrong dimensions");
	}

	Tensor result(rows, cols);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			float value = get(i, j) + t.get(i, j);
			result.set(i, j, value);
		}
	}

	return result;
}

Tensor Tensor::operator*(float s) const
{
	Tensor result(rows, cols);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			float value = get(i, j) * s;
			result.set(i, j, value);
		}
	}

	return result;
}

Tensor Tensor::operator+(float s) const
{
	Tensor result(rows, cols);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			float value = get(i, j) + s;
			result.set(i, j, value);
		}
	}

	return result;
}

Tensor Tensor::operator-(float s) const
{
	Tensor result(rows, cols);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			float value = get(i, j) - s;
			result.set(i, j, value);
		}
	}

	return result;
}

Tensor Tensor::operator/(float s) const
{
	if (s == 0.0f) {
		throw std::invalid_argument("Cannot divide by zero.");
	}

	Tensor result(rows, cols);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			float value = get(i, j) / s;
			result.set(i, j, value);
		}
	}

	return result;
}

Tensor operator*(float s, const Tensor& t) {
	return t * s;
}

Tensor operator+(float s, const Tensor& t) {
	return t + s;
}