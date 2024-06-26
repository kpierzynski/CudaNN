#include "Loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <iostream>

Loader::Loader(const std::string& images_path, int batch_size, int cnt)
{
	int w, h, n;

	std::filesystem::path root(images_path);

	std::vector<std::vector<float>> all_images;
	std::vector<int> all_labels;

	for (const auto& item : std::filesystem::directory_iterator(root)) {
		if (std::filesystem::is_directory(item)) {
			std::string label = item.path().filename().string();
			int i_label = std::stoi(label);

			std::cout << "LABEL (" << label << "): " << i_label << std::endl;

			int class_cnt = 0;
			for (const auto& image : std::filesystem::directory_iterator(item)) {
				std::string image_path = image.path().string();

				uint8_t* image_data = stbi_load(image_path.c_str(), &w, &h, &n, 1);

				if (w != 128 || h != 128 || n != 1) {
					printf("ABORT: error while reading plant images. Read dims: (%d, %d, %d)\r\n", w, h, n);
					exit(-1);
				}

				std::vector<float> f_image_data;

				for (int i = 0; i < w * h * n; i++) {
					f_image_data.push_back((float)( image_data[i] ) / 255.0f);
				}

				all_images.push_back(f_image_data);
				all_labels.push_back(i_label);

				class_cnt++;
				if (class_cnt >= cnt) break;
			}
		}
	}

	item_count = all_images.size();

	auto seed = time(0);

	std::mt19937 gen(seed);
	std::shuffle(all_images.begin(), all_images.end(), gen);

	std::mt19937 gen2(seed);
	std::shuffle(all_labels.begin(), all_labels.end(), gen2);


	for (int i = 0; i < all_images.size() / batch_size; i++) {
		std::vector<float> batch;
		std::vector<float> labels_batch;

		for (int b = 0; b < batch_size; b++) {
			batch.insert(batch.end(), all_images[i * batch_size + b].begin(), all_images[i * batch_size + b].end());
			
			for (int c = 0; c < 10; c++) {
				labels_batch.push_back((c == all_labels[i * batch_size + b]) ? 1.0f : 0.0f);
			}
		}

		if (batch.size() != batch_size * w * h * n) {
			printf("Error in size. Aborting... s: %lld,  (%d %d %d %d)\r\n", batch.size(), batch_size, w, h, n);
			exit(-1);
		}

		this->images.push_back(new Tensor(batch_size, w * h * n, batch.data()));
		this->labels.push_back(new Tensor(batch_size, 10, labels_batch.data()));
	}
}

Loader::~Loader()
{
}

void Loader::print(int index, int batch) {
	std::cout << "Image at position: " << index << " has label: " << std::endl;
	(*labels[index]).print();

	const char asciiChars[] = " .:-=+*#%@";
	for (int y = 0; y < 64; ++y) {
		for (int x = 0; x < 64; ++x) {
			float pixel = (*images[index])[(y * 64 + x)] * 255;
			float charIndex = pixel / 26;
			std::cout << asciiChars[(int)charIndex];
		}
		std::cout << std::endl;
	}
}