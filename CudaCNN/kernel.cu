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

/*__global__ void linearLayerForward1(float* W, float* A, float* Z, float* b,
								   int W_x_dim, int W_y_dim,
								   int A_x_dim, int A_y_dim) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < A_x_dim && col < W_y_dim) {
		float sum = 0.0f;
		for (int i = 0; i < A_y_dim; i++) {
			sum += A[row * A_y_dim + i] * W[i * W_y_dim + col];
		}
		Z[row * W_y_dim + col] = sum + b[col];
	}
}*/

int main()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return -1;
	}

	/*

	float input[12] = { 1,2,3,4, 5,6,7,8, 1,0,1,2 };
	Tensor t_input(3, 4);
	t_input.set_from(input, 12);
	t_input.print();

	float weights[8] = { 1, 0, 2, 0, 1, 1, 3, 2 };
	Tensor t_weights(4, 2);
	t_weights.set_from(weights, 8);
	t_weights.print();

	float biases[2] = { 0.1f, -0.1f };
	Tensor t_biases(1, 2);
	t_biases.set_from(biases, 2);
	t_biases.print();

	float output[6];

	float* dev_input;
	float* dev_weights;
	float* dev_biases;
	float* dev_output;

	cudaMalloc((void**)&dev_input, 12 * sizeof(float));
	cudaMemcpy(dev_input, input, 12 * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_weights, 8 * sizeof(float));
	cudaMemcpy(dev_weights, weights, 8 * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_output, 6 * sizeof(float));
	cudaMemcpy(dev_output, output, 6 * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_biases, 2 * sizeof(float));
	cudaMemcpy(dev_biases, biases, 2 * sizeof(float), cudaMemcpyHostToDevice);

	dim3 block_size(1, 1);

	dim3 num_of_blocks((3 + block_size.x) / block_size.x,
					   (2 + block_size.y) / block_size.y);

	linearLayerForward1 << <num_of_blocks, block_size >> > (dev_weights, dev_input, dev_output, dev_biases,  4, 2, 3, 4 );

	cudaMemcpy(output, dev_output, 6 * sizeof(float), cudaMemcpyDeviceToHost);

	Tensor t_output(3, 2);
	t_output.set_from(output, 6);
	t_output.print();
	*/

	Network net;
	net.addLayer(new Linear(28 * 28, 30));
	net.addLayer(new Tanh(30));
	net.addLayer(new Linear(30, 10));
	net.addLayer(new Tanh(10));

	MNISTSet mnist(std::string("C:\\MNIST\\train-images.idx3-ubyte"), std::string("C:\\MNIST\\train-labels.idx1-ubyte"), 8, 60000);
	mnist.print(0, 0);

	std::cout << "Fitting" << std::endl;
	net.fit(mnist.images, mnist.labels, 0.01f, 5);

	MNISTSet mnist_test(std::string("C:\\MNIST\\t10k-images.idx3-ubyte"), std::string("C:\\MNIST\\t10k-labels.idx1-ubyte"), 8, 10000);
	net.evaluate(mnist_test.images, mnist_test.labels);

	return 0;
}
