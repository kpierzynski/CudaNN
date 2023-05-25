#pragma once

#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>

#include "Tensor.h"

#define DEBUG 1

class MNISTSet {
public:

	std::vector<Tensor> labels;
	std::vector<Tensor> images;

	int item_count;

	MNISTSet(const std::string& images_path, const std::string& labels_path, int cnt);
	~MNISTSet();

	void print(int index);

private:
	int reverse(int value);
};