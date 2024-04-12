#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "Mnist.h"
#include "Network.h"
#include "Linear.h"
#include "MSE.h"
#include "Tanh.h"
#include "Loader.h"

#define BATCH_SIZE 32

int main()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("Error! Cannot set device 0. Aborting.\r\n");
		exit(-1);
	}

	Mnist mnist(std::string("../data/train-images-idx3-ubyte"), std::string("../data/train-labels-idx1-ubyte"), BATCH_SIZE, 300000);
	mnist.print(1, 0);

	Network net;
	net.addLayer(new Linear(28 * 28, 30, BATCH_SIZE));
	net.addLayer(new Tanh(30, BATCH_SIZE));
	net.addLayer(new Linear(30, 10, BATCH_SIZE));
	net.addLayer(new Tanh(10, BATCH_SIZE));

	net.fit(mnist.images, mnist.labels, 0.01f, 10);

	Mnist mnist_test(std::string("../data/t10k-images-idx3-ubyte"), std::string("../data/t10k-labels-idx1-ubyte"), BATCH_SIZE, 10000);
	net.evaluate(mnist_test.images, mnist_test.labels);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		printf("Cannot reset cuda device. Aborting.\r\n");
		exit(-1);
	}

	return 0;
}

