
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Mnist.h"
#include "Network.h"
#include "Linear.h"
#include "MSE.h"

#include <stdio.h>

#define BATCH_SIZE 8

int main()
{
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 1;
    }

    // CODE HERE

    #if 1 == 0
    float v1[] = { 1111.0f, 1.1f, -1.0f, 11.0f, -5.5f, 4.2f, 3.76f, 8.88f, 111.0f, -1000.5f, 14.34f, 0.64f };
    float v2[] = { 1.0f, 1.1f, -1.0f, 11.0f, -5.5f, 4.2f, 3.76f, 8.88f, 111.0f, 1000.5f, 14.34f, 0.64f };
    Tensor t1(3, 4, v1);
    Tensor t2(3, 4, v2);
    t1.print();
    t2.print();

    convert(t1);
    convert(t2);

    t1.print();
    t2.print();

    int r = compare(t1, t2);
    std::cout << r << std::endl;

    return -1;
    #endif

    #if 1 == 0
    //  MSE LOSS TEST
    float v1[] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.5f, 1.0f, 0.6f };
    float v2[] = { -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -0.6f };

    Tensor t1(3, 4, v1);
    Tensor t2(3, 4, v2);

    t1.print();
    t2.print();

    float l = MSE::cost(t1, t2);
    std::cout << l << std::endl;

    return -1;
    #endif

    #if 1 == 0
    //  MSE DERIV TEST
    float v1[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.5f, 1.0f, 0.6f};
    float v2[] = { -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -0.6f };

    Tensor t1(3, 4, v1);
    Tensor t2(3, 4, v2);

    t1.print();
    t2.print();

    Tensor * deriv = MSE::derivative(t1, t2);
    deriv->dev2host();
    deriv->print();

    return -1;
    #endif

    Mnist mnist(std::string("C:\\MNIST\\train-images.idx3-ubyte"), std::string("C:\\MNIST\\train-labels.idx1-ubyte"), BATCH_SIZE, 60000);
    mnist.print(1, 0);
    
    Network net;
    net.addLayer(new Linear(28*28, 10, BATCH_SIZE));

    net.fit(mnist.images, mnist.labels, 0.01, 5);
    
    Mnist mnist_test(std::string("C:\\MNIST\\t10k-images.idx3-ubyte"), std::string("C:\\MNIST\\t10k-labels.idx1-ubyte"), BATCH_SIZE, 10000);
    net.evaluate(mnist_test.images, mnist_test.labels);
    // CODE UP THERE

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

