
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "Tensor.h"
#include "Network.h"
#include "Linear.h"
#include "MNISTSet.h"


int main()
{
	MNISTSet mnist(std::string("C:\\MNIST\\train-images.idx3-ubyte"), std::string("C:\\MNIST\\train-labels.idx1-ubyte"));
	mnist.print(0);

	Layer* p = new Linear(28 * 28, 10);

	Network net;
	net.addLayer(p);

	net.fit(mnist.images, mnist.labels, 0.01, 3);

	return 0;
}
