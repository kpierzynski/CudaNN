
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
		printf("Error! Cannot set device 0. Aborting.\r\n");
		exit(-1);
	}

	//#define MNIST

	#ifdef MNIST
	Plants plants("C:\\IMAGE_MNIST_RESIZED", BATCH_SIZE, 3500);

	Network net2;
	net2.addLayer(new Linear(256 * 256 * 1, 32, BATCH_SIZE));
	net2.addLayer(new Tanh(32, BATCH_SIZE));

	net2.addLayer(new Linear(32, 10, BATCH_SIZE));
	net2.addLayer(new Tanh(10, BATCH_SIZE));

	net2.fit(plants.images, plants.labels, 0.01f, 10);

	//Plants plants_test("C:\\PLANTS\\test", BATCH_SIZE, 72);
	net2.evaluate(plants.images, plants.labels);

	#else

	Mnist mnist(std::string("C:\\EMNIST\\emnist-digits-train-images-idx3-ubyte"), std::string("C:\\EMNIST\\emnist-digits-train-labels-idx1-ubyte"), BATCH_SIZE, 300000);
	mnist.print(1, 0);

	Network net;
	net.addLayer(new Linear(28 * 28, 30, BATCH_SIZE));
	net.addLayer(new Tanh(30, BATCH_SIZE));
	net.addLayer(new Linear(30, 10, BATCH_SIZE));
	net.addLayer(new Tanh(10, BATCH_SIZE));

	net.fit(mnist.images, mnist.labels, 0.01f, 10);

	Mnist mnist_test(std::string("C:\\EMNIST\\emnist-digits-test-images-idx3-ubyte"), std::string("C:\\EMNIST\\emnist-digits-test-labels-idx1-ubyte"), BATCH_SIZE, 10000);
	net.evaluate(mnist_test.images, mnist_test.labels);
	#endif

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		printf("Cannot reset cuda device. Aborting.\r\n");
		exit(-1);
	}

	return 0;
}

