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

__global__ void linearLayerForward(float* W, float* A, float* Z, float* b,
                                   int W_x_dim, int W_y_dim,
                                   int A_x_dim, int A_y_dim) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int Z_x_dim = A_x_dim;
    int Z_y_dim = W_y_dim;

    float Z_value = 0;

    if (row < Z_y_dim && col < Z_x_dim) {
        for (int i = 0; i < W_x_dim; i++) {
            Z_value += W[row * W_x_dim + i] * A[i * A_x_dim + col];
        }
        Z[row * Z_x_dim + col] = Z_value + b[row];
    }
}

int main()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return -1;
	}

	float input[] = {1,2,3,4, 5,6,7,8, 9,0,-1,1.5f};
	float weights[] = { 0.5f, 1.0f, -1.0f, 0.5f, -0.25f, 0.25f, -0.5f, 0 };

	float biases[] = { 0.1f, -0.1f };
	float output[3*2];

	float* dev_input;
	float* dev_weights;
	float* dev_biases;
	float* dev_output;

	cudaMalloc((void**)&dev_weights, 8 * sizeof(float));
	cudaMemcpy(dev_weights, weights, 8 * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_input, 12 * sizeof(float));
	cudaMemcpy(dev_input, input, 12 * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_output, 6 * sizeof(float));
	cudaMemcpy(dev_output, output, 6 * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_biases, 2 * sizeof(float));
	cudaMemcpy(dev_biases, biases, 2 * sizeof(float), cudaMemcpyHostToDevice);

	dim3 block_size(1, 1);

	dim3 num_of_blocks((3 + block_size.x - 1) / block_size.x,
					   (2 + block_size.y - 1) / block_size.y);

	linearLayerForward << <num_of_blocks, block_size >> > (dev_weights, dev_input, dev_output, dev_biases, 4, 2, 3, 4);

	cudaMemcpy(output, dev_output, 6 * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 2; j++) {
			std::cout << output[i * 2 + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	return -1;
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
