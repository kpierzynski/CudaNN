from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        pass

    def __getitem__(self, idx: int) -> tuple[(torch.Tensor | np.ndarray), int]:  # TODO needs to decide image type
        image, label = None
        return image, label

    def read_images(self, file_path: str) -> np.ndarray:
        file_path = Path(file_path)
        with open(file_path, "rb") as f:
            _ = int.from_bytes(f.read(4), "big")
            num_images = int.from_bytes(f.read(4), "big")
            num_rows = int.from_bytes(f.read(4), "big")
            num_cols = int.from_bytes(f.read(4), "big")
            images_raw = f.read()

        images = np.frombuffer(images_raw, dtype=np.uint8)
        images = images.reshape(num_images, num_rows, num_cols)
        return images

    def read_labels(self, file_path: str) -> np.ndarray:
        file_path = Path(file_path)
        with open(file_path, "rb") as f:
            _ = int.from_bytes(f.read(4), "big")
            _ = int.from_bytes(f.read(4), "big")
            labels_raw = f.read()

        labels = np.frombuffer(labels_raw, dtype=np.uint8)
        return labels

    def display_images(self, images, labels, save_example: bool = False):
        fig, axs = plt.subplots(3, 5, figsize=(10, 6))
        for i, ax in enumerate(axs.flat):
            ax.imshow(images[i], cmap="gray")
            ax.axis("off")
            ax.set_title(f"Label: {labels[i]}")
        if save_example:
            fig.savefig("data/example.jpg")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    images_file_path = "data/train-images-idx3-ubyte"
    labels_file_path = "data/train-labels-idx1-ubyte"

    dataset = Dataset()

    images = dataset.read_images(images_file_path)
    labels = dataset.read_labels(labels_file_path)

    dataset.display_images(images, labels)
