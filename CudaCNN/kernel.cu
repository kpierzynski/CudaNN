﻿
#pragma float_control( except, on )

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "Tensor.h"
#include "Network.h"
#include "Linear.h"
#include "Tanh.h"
#include "ReLU.h"
#include "MNISTSet.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main()
{
	Network net;
	net.addLayer(new Linear(28 * 28, 128));
	net.addLayer(new Tanh(128));
	net.addLayer(new Linear(128, 10));
	net.addLayer(new Tanh(10));

	MNISTSet mnist(std::string("C:\\MNIST\\train-images.idx3-ubyte"), std::string("C:\\MNIST\\train-labels.idx1-ubyte"), 8, 60000);
	mnist.print(0, 0);

	std::cout << "Fitting" << std::endl;
	net.fit(mnist.images, mnist.labels, 0.01f, 10);

	MNISTSet mnist_test(std::string("C:\\MNIST\\t10k-images.idx3-ubyte"), std::string("C:\\MNIST\\t10k-labels.idx1-ubyte"), 8, 10000);
	net.evaluate(mnist_test.images, mnist_test.labels);

	return 0;
}
