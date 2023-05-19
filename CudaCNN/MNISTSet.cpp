#include "MNISTSet.h"

MNISTSet::MNISTSet(const std::string& images_path, const std::string& labels_path) {
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
		std::vector<float> f_image_data;

		uint8_t label_data;

		images_file.read((char*)image_data, w * h);
		labels_file.read((char*)&label_data, 1);

		for (int i = 0; i < w * h; i++) f_image_data.push_back(image_data[i] / 255.0f);

		images.push_back(f_image_data);
		labels.push_back(label_data);

		delete[] image_data;
	}

#if DEBUG == 1
	std::cout << "Done reading images" << std::endl;
#endif

	item_count = images_count;

	images_file.close();
	labels_file.close();
}

void MNISTSet::print(int index) {
	std::cout << "Image at position: " << index << " has label: " << labels[index] << std::endl;
		const char asciiChars[] = " .:-=+*#%@";
		for (int y = 0; y < 28; ++y) {
		for (int x = 0; x < 28; ++x) {
			float pixel = images[index][y * 28 + x] * 255;
			float charIndex = pixel / 26;
				std::cout << asciiChars[(int)charIndex];
		}
		std::cout << std::endl;
	}
}

MNISTSet::~MNISTSet() {

}

int MNISTSet::reverse(int value) {
	uint32_t result = ((value & 0xFF000000) >> 24) |
		((value & 0x00FF0000) >> 8) |
		((value & 0x0000FF00) << 8) |
		((value & 0x000000FF) << 24);
	return static_cast<uint32_t>(result);
}
