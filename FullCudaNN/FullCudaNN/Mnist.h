#pragma once

#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <random>

#include "Tensor.h"

#define DEBUG 1

class Mnist {
	public:

	std::vector<Tensor*> labels;
	std::vector<Tensor*> images;

	int item_count;

	Mnist(const std::string& images_path, const std::string& labels_path, int batch_size, int cnt);
	~Mnist();

	void print(int index, int batch);

	private:
	int reverse(int value);
};