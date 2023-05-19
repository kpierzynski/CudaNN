#pragma once

#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>

#define DEBUG 1

class MNISTSet {
public:

	std::vector<int> labels;
	std::vector<float*> images;

	int item_count;

	MNISTSet(const std::string& images_path, const std::string& labels_path);
	~MNISTSet();

	void print(int index);

private:
	int reverse(int value);
};