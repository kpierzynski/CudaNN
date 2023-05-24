#include "Tensor.h"

Tensor::Tensor(int rows, int cols) : rows(rows), cols(cols) {
	data = new float[rows * cols];
	set_random();
}

Tensor::Tensor(const Tensor& t) : rows(t.rows), cols(t.cols) {
	data = new float[rows * cols];

	std::copy(t.data, t.data + (rows * cols), data);
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

void Tensor::set_random()
{
	std::random_device dev;
	std::mt19937 gen(dev());

	std::uniform_real_distribution<float> unif(-0.5, 0.5);

	for (int i = 0; i < rows * cols; i++) {
		data[i] = unif(gen);
	}
}

void Tensor::print() 
{
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			std::cout << data[r * cols + c] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

int Tensor::size() const
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

float Tensor::mean() const
{
	float sum = 0.0f;
	for (int i = 0; i < this->size(); i++) {
		sum += data[i];
	}

	return sum / size();
}

Tensor Tensor::transpose() const
{
	Tensor result(cols, rows);

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			result.set(j, i, get(i, j));
		}
	}

	return result;
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
	if (cols == t.cols && rows == t.rows) {
		Tensor result(rows, cols);

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				float value = get(i, j) * t.get(i, j);
				result.set(i, j, value);
			}
		}

		return result;
	}
	else if (cols == t.rows) {
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
	} else {
		throw std::invalid_argument("Cannot perform matrix multiplcation. Wrong dimentions");
	}	
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

Tensor& Tensor::operator+=(const Tensor& t)
{
	if (rows != t.rows || cols != t.cols) {
		throw std::invalid_argument("Cannot add tensor. Wrong dimensions.");
	}

	for (int i = 0; i < rows * cols; i++) {
		data[i] += t.data[i];
	}

	return *this;
}

Tensor& Tensor::operator-=(const Tensor& t)
{
	if (rows != t.rows || cols != t.cols) {
		throw std::invalid_argument("Cannot subtract tensor. Wrong dimensions.");
	}

	for (int i = 0; i < rows * cols; i++) {
		data[i] -= t.data[i];
	}

	return *this;
}

Tensor& Tensor::operator=(const Tensor& t)
{
	if (this == &t) {
		return *this;
	}

	delete[] data;

	rows = t.rows;
	cols = t.cols;

	data = new float[rows * cols];

	std::copy(t.data, t.data + (rows * cols), data);

	return *this;
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

std::ostream& operator<<(std::ostream& stream, const Tensor& t)
{
	for (int r = 0; r < t.rows; r++) {
		for (int c = 0; c < t.cols; c++) {
			stream << t.data[r * t.cols + c] << " ";
		}
		stream << std::endl;
	}
	stream << std::endl;

	return stream;
}
