#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Tensor.h"

class MSE {
	public:
	static float cost(Tensor& y_pred, Tensor& y_real);
	static void derivative(Tensor& result, Tensor& y_pred, Tensor& y_real);
};