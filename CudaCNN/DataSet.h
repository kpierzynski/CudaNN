#pragma once
#include <cstdio>
#include <string>
#include <vector>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define DEBUG 1

// image loader candicates:
// - https://github.com/nothings/stb
// - https://github.com/charlesdong/tiny-image/blob/master/src/tinyimage.cpp

// using c++17
// - https://stackoverflow.com/questions/63036624/how-to-enable-c17-code-generation-in-vs2019-cuda-project

/* root
	- train
		- 0
		- 1
		- ..
	- test
		- 0
			- img.jpg
		- 1
			- img1.jpg
*/

typedef struct {
	int label;
	uint8_t* ptr;
} item_t;

/*
	- załadować ze ścieżki obrazki do pamięci
	- móc zwrócić jeden obrazek i batch obrazków (+labelki)
*/
class DataSet {

public:
	const std::string path;

	int size = 0;	// wielkosc calego datasetu w bajtach
	int item_count = 0; // ilosc obrazkow

	int batch_size;

	std::vector<int> labels;
	std::vector<uint8_t*> images;

	uint8_t* host_data;
	uint8_t* device_data;

	DataSet(const std::string& path, int batch_size) : batch_size(batch_size) {

		std::filesystem::path root(path);

		for (const auto& item : std::filesystem::directory_iterator(root)) {
			if (std::filesystem::is_directory(item)) {
				std::string label = item.path().filename().string();
				int i_label = std::stoi(label);

				for (const auto& image : std::filesystem::directory_iterator(item)) {
					std::string image_path = image.path().string();

					int w, h, n;
					uint8_t* image_data = stbi_load(image_path.c_str(), &w, &h, &n, STBI_rgb);

					labels.push_back(i_label);
					images.push_back(image_data);

					item_count++;
					size += w * h * n;

#if DEBUG == 1
					std::cout << "(w,h,n): (" << w << ',' << h << ',' << n << ')' << std::endl;
#endif
				}

			}
		}

#if DEBUG == 1
		std::cout << "itms: " << item_count << " size: " << size << std::endl;

		for (int label : labels) {
			std::cout << label << std::endl;
		}
#endif
	}

	~DataSet() {
		for (uint8_t* ptr : images) {
			stbi_image_free(ptr);
		}
	}

	item_t get(int index) {

	}

	item_t* get_batch(int batch_index) {

	}

};