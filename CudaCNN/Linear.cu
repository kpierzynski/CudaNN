#include "Linear.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void linearLayerForward(float* W, float* A, float* Z, float* b,
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
}

Linear::Linear(int input_size, int output_size) :
	Layer(input_size, output_size),
	weights(Tensor(input_size, output_size)),
	biases(Tensor(1, output_size)),
	input(Tensor(1, input_size)),
	output(Tensor(8, output_size))
{
	//weights.set_random();
	//biases.set_random();

	float* dev_weights = 0;
	cudaMalloc((void**)&dev_weights, weights.size() * sizeof(int));
	cudaMemcpy(dev_weights, weights.data, weights.size() * sizeof(int), cudaMemcpyHostToDevice);

	float* dev_input = 0;
	cudaMalloc((void**)&dev_input, input.size() * sizeof(int));
	cudaMemcpy(dev_input, input.data, input.size() * sizeof(int), cudaMemcpyHostToDevice);

	float* dev_output = 0;
	cudaMalloc((void**)&dev_output, output.size() * sizeof(int));
	cudaMemcpy(dev_output, output.data, output.size() * sizeof(int), cudaMemcpyHostToDevice);

	float* dev_biases = 0;
	cudaMalloc((void**)&dev_biases, biases.size() * sizeof(int));
	cudaMemcpy(dev_biases, biases.data, biases.size() * sizeof(int), cudaMemcpyHostToDevice);
}

Tensor Linear::forward(Tensor& input)
{
	this->input = input;
	
	//Tensor output(input.rows, weights.cols);

	dim3 block_size(1, 1);


	dim3 num_of_blocks((input.rows + block_size.x) / block_size.x,
					   (weights.cols + block_size.y) / block_size.y);
	
	linearLayerForward<<<num_of_blocks, block_size>>>(dev_weights, dev_input, dev_output, dev_biases, weights.rows, weights.cols, input.rows, input.cols);

	cudaMemcpy(output.data, dev_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost);

	//output.print();
	
	//CPU
	//Tensor output = input * weights;
	//output += biases;

	return output;
}

Tensor Linear::backward(Tensor& gradient, float lr)
{
	Tensor wGradient = input.transpose() * gradient;
	Tensor bGradient = gradient;

	weights -= wGradient * lr;
	biases -= (bGradient * lr).sum_rows();

	Tensor output = gradient * weights.transpose();

	return output;
}

