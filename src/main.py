import torch
from tqdm import tqdm

from dataset import Dataset
from model import MLPModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
INITIAL_LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 10

torch.cuda.empty_cache()


def main():
    train_dataset = Dataset(
        images_file_path="data/t10k-images-idx3-ubyte",
        labels_file_path="data/t10k-labels-idx1-ubyte",
    )
    test_dataset = Dataset(
        images_file_path="data/train-images-idx3-ubyte",
        labels_file_path="data/train-labels-idx1-ubyte",
    )

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MLPModel(in_features=28 * 28).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=INITIAL_LR)
    criterion = torch.nn.CrossEntropyLoss()

    print("[INFO]\t start training loop...")
    for epoch in tqdm(range(EPOCHS)):
        epoch_loss = 0
        epoch_accuracy = 0

        for batch, (image, label) in enumerate(train_dataloader):
            image = image.to(DEVICE)
            # print(label, end=" ")
            label = label.to(DEVICE)
            prediction = model(image)
            loss = criterion(prediction, label)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_accuracy += (prediction.argmax(dim=1) == label).sum().item()

        train_loss = epoch_loss / len(train_dataloader)
        train_accuracy = epoch_accuracy / len(train_dataset)

        tqdm.write(f"Epoch {epoch+1}/{EPOCHS} Train loss: {train_loss:.4f}| train accuracy: {train_accuracy:.4f}")

    print(f"[INFO]\t Total train loss: {train_loss:.4f}|Total train accuracy: {train_accuracy:.4f}")

    print("[INFO]\t finished training loop")
    print("[INFO]\t start testing loop...")

    test_loss = 0
    test_accuracy = 0

    # evaluation loop
    with torch.no_grad():
        for batch, (image, label) in enumerate(tqdm(test_dataloader)):
            image = image.to(DEVICE)
            label = label.to(DEVICE)

            prediction = model(image)
            loss = criterion(prediction, label)

            epoch_loss += loss.item()
            epoch_accuracy += (prediction.argmax(dim=1) == label).sum().item()

        test_loss = epoch_loss / len(test_dataloader)
        test_accuracy = epoch_accuracy / len(test_dataset)

        print(f"[INFO]\t Test loss: {test_loss:.4f}| test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
