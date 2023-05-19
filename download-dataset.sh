#!/bin/sh

mkdir -p data

#train data
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P data && gzip -d data/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P data && gzip -d data/train-labels-idx1-ubyte.gz
#test data
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -P data && gzip -d data/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -P data && gzip -d data/t10k-labels-idx1-ubyte.gz