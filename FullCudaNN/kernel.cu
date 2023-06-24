
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "Mnist.h"
#include "Network.h"
#include "Linear.h"
#include "MSE.h"
#include "Tanh.h"
#include "Plants.h"

#define BATCH_SIZE 32

int main()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\r\n");
		return 1;
	}

	Plants plants("C:\\PLANTS\\color", BATCH_SIZE, 360);
	Network net2;
	net2.addLayer(new Linear(256 * 256 * 3, 512, BATCH_SIZE));
	net2.addLayer(new Tanh(512, BATCH_SIZE));

	net2.addLayer(new Linear(512, 128, BATCH_SIZE));
	net2.addLayer(new Tanh(128, BATCH_SIZE));

	net2.addLayer(new Linear(128, 10, BATCH_SIZE));
	net2.addLayer(new Tanh(10, BATCH_SIZE));

	net2.fit(plants.images, plants.labels, 0.001, 3);

	return -1;

	Mnist mnist(std::string("C:\\MNIST\\train-images.idx3-ubyte"), std::string("C:\\MNIST\\train-labels.idx1-ubyte"), BATCH_SIZE, 60000);
	mnist.print(1, 0);

	Network net;
	net.addLayer(new Linear(28 * 28, 128, BATCH_SIZE));
	net.addLayer(new Tanh(128, BATCH_SIZE));
	net.addLayer(new Linear(128, 10, BATCH_SIZE));
	net.addLayer(new Tanh(10, BATCH_SIZE));

	net.fit(mnist.images, mnist.labels, 0.01, 10);

	Mnist mnist_test(std::string("C:\\MNIST\\t10k-images.idx3-ubyte"), std::string("C:\\MNIST\\t10k-labels.idx1-ubyte"), BATCH_SIZE, 10000);
	net.evaluate(mnist_test.images, mnist_test.labels);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

