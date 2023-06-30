#pragma once

#include <vector>
#include <string>
#include <filesystem>
#include <random>

#include "Tensor.h"

class Loader {
	public:
	
	int item_count;

	std::vector<Tensor*> images;
	std::vector<Tensor*> labels;

	Loader(const std::string& images_path, int batch_size, int cnt);
	~Loader();
	void print(int index, int batch);
};