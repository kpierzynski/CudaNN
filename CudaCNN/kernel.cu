
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "Tensor.h"


int main()
{

	Tensor p(2, 4);
	p.set_from({ 1,2,3,4,0,-3,2.5,1.3 });
	p.print();

	Tensor r(4, 3);
	r.set_from({ 1,-1.5,0, 2.1,2,0, -4,3,7, -1,0.5,-2 });
	r.print();

	(p * r).print();



	return 0;
}
