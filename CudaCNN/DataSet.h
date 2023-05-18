#pragma once
#include <string>
#include <vector>

// image loader candicates:
// - https://github.com/nothings/stb
// - https://github.com/charlesdong/tiny-image/blob/master/src/tinyimage.cpp

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

	const int size;	// wielkosc calego datasetu w bajtach
	const int item_count; // ilosc obrazkow

	const int batch_size;

	std::vector<int> pos; // pozycje na ktorych zaczyna sie para (labelka,dane obrazka)

	const uint8_t* host_data;
	const uint8_t* device_data;

	DataSet(const std::string & path, int batch_size) : batch_size(batch_size) {
		// przejść path
		// policzyc obrazki
		// malloc() -> new []
		// cudaMalloc()
		// 0 0xF3 .. 0xFF 7 0x7E ..
	}

	~Dataset() {
		// delete
		// cudaFree
	}

	item_t get(int index) {
		item_t item = {
			.label = host_data[this->pos[index]],
			.ptr = &device_data[this->pos[index] + 1]
		};

		return item;
	}

	item_t * get_batch(int batch_index) {
		// this->batch_size
	}
};