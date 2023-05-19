Python implementation for MLP neural network
====

# Dataset
This project utilizes the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, which consists of handwritten digit images, where each image is a grayscale 28x28 pixel representation of a digit from 0 to 9.

# How to run

Create conda enviroment using following command:
```bash
conda env create -f env.yml
```

Download dataset by executing script:
```bash
./download-dataset.sh
```

For train and test type below command:
```bash 
python src/main.py
```