#pragma once

#include <vector>
#include <string>
#include <filesystem>
#include <random>

#include "Tensor.h"

class Plants {
	public:
	
	int item_count;

	std::vector<Tensor*> images;
	std::vector<Tensor*> labels;

	Plants(const std::string& images_path, int batch_size, int cnt);
	~Plants();
	void print(int index, int batch);
};