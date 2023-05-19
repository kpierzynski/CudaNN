#include "DataSet.h"

DataSet::DataSet(const std::string& path, int batch_size) : batch_size(batch_size) {
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

DataSet::~DataSet() {
	for (uint8_t* ptr : images) {
		stbi_image_free(ptr);
	}
}

item_t DataSet::get(int index) {

}

item_t* DataSet::get_batch(int batch_index) {

}