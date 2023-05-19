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

class MNIST {
public:

	std::vector<int> labels;
	std::vector<uint8_t*> images;

	int item_count;

	MNIST(const std::string& images_path, const std::string& labels_path) {
		std::ifstream images_file(images_path, std::ios::binary);
		std::ifstream labels_file(labels_path, std::ios::binary);

		if (!images_file || !labels_file) {
			throw std::runtime_error("Cannot open images file.");
		}

		images_file.seekg(0, images_file.beg);

		int images_magic_number = 0, images_count, w, h;
		int labels_magic_number = 0, labels_count;

		images_file.read((char*)&images_magic_number, 4);
		labels_file.read((char*)&labels_magic_number, 4);
		images_magic_number = reverse(images_magic_number);
		labels_magic_number = reverse(labels_magic_number);

		if (images_magic_number != 0x803 || labels_magic_number != 0x801) {
			throw std::runtime_error("Wrong magic_number");
		}

		images_file.read((char*)&images_count, 4);
		images_count = reverse(images_count);

		images_file.read((char*)&w, 4);
		images_file.read((char*)&h, 4);
		w = reverse(w);
		h = reverse(h);

		labels_file.read((char*)&labels_count, 4);
		labels_count = reverse(labels_count);

		if (images_count != labels_count) {
			throw std::runtime_error("Wrong item amount between labels and data.");
		}

#if DEBUG == 1
		std::cout << "Start reading images, #" << images_count << " (" << w << "," << h << ")" << std::endl;
#endif

		for (int i = 0; i < images_count; i++) {
			uint8_t* image_data = new uint8_t[w * h];
			uint8_t label_data;

			images_file.read((char*)image_data, w * h);
			labels_file.read((char*)&label_data, 1);

			images.push_back(image_data);
			labels.push_back(label_data);
		}

#if DEBUG == 1
		std::cout << "Done reading images" << std::endl;
#endif

		item_count = images_count;

		images_file.close();
		labels_file.close();
	}

	void print(int index) {
		std::cout << "Image at position: " << index << " has label: " << labels[index] << std::endl;

		const char asciiChars[] = " .:-=+*#%@";

		for (int y = 0; y < 28; ++y) {
			for (int x = 0; x < 28; ++x) {
				uint8_t pixel = images[index][y * 28 + x];
				uint8_t charIndex = pixel / 26;

				std::cout << asciiChars[charIndex];
			}
			std::cout << std::endl;
		}
	}

	~MNIST() {
		for (const auto& data : images) {
			delete[] data;
		}
	}

private:
	int reverse(int value) {
		uint32_t result = ((value & 0xFF000000) >> 24) |
			((value & 0x00FF0000) >> 8) |
			((value & 0x0000FF00) << 8) |
			((value & 0x000000FF) << 24);
		return static_cast<uint32_t>(result);
	}
};

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