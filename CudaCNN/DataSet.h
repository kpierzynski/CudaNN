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

	DataSet(const std::string& path, int batch_size);
	~DataSet();

	item_t get(int index);
	item_t* get_batch(int batch_index);
};