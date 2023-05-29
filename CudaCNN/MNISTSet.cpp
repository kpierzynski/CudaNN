#include "MNISTSet.h"

MNISTSet::MNISTSet(const std::string& images_path, const std::string& labels_path, int batch_size, int cnt) {
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

	item_count = (cnt < images_count) ? cnt : images_count;

	for (int b = 0; b < item_count / batch_size; b++) {
		Tensor t_label(batch_size, 10);
		Tensor t_image(batch_size, w * h);

		std::vector<float> f_image_data;
		std::vector<float> f_label;

		for (int i = 0; i < batch_size; i++) {
			uint8_t* image_data = new uint8_t[w * h];

			uint8_t label_data;

			images_file.read((char*)image_data, w * h);
			labels_file.read((char*)&label_data, 1);

			for (int k = 0; k < w * h; k++) f_image_data.push_back(image_data[k] / 255.0f);
			for (int k = 0; k < 10; k++) {
				f_label.push_back((k == label_data) ? 1 : 0);
			}

			delete[] image_data;
		}
		t_image.set_from(f_image_data);
		t_label.set_from(f_label);

		images.push_back(t_image);
		labels.push_back(t_label);
	}

	#if DEBUG == 1
	std::cout << "Done reading images" << std::endl;
	#endif

	auto seed = unsigned(std::time(0));

	std::srand(seed);
	std::random_shuffle(images.begin(), images.end());

	std::srand(seed);
	std::random_shuffle(labels.begin(), labels.end());

	images_file.close();
	labels_file.close();
}

void MNISTSet::print(int index, int batch) {
	std::cout << "Image at position: " << index << " has label: " << labels[index][batch] << std::endl;
	const char asciiChars[] = " .:-=+*#%@";
	for (int y = 0; y < 28; ++y) {
		for (int x = 0; x < 28; ++x) {
			float pixel = images[index].get(batch, y * 28 + x) * 255;
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
