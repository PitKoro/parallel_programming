#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CSC(call)								\
	do {										\
		cudaError_t status = call;				\
		if (status != cudaSuccess){				\
			fprintf(stderr, "ERROR in %s:%d. Message: %s\n", __FILE__, __LINE__,	\
				cudaGetErrorString(status));	\
			exit(0);							\
		}										\
	} while(0)									\

__global__ void reverse_vector(double* a, int l) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	while (idx < ceilf(l / 2)) {
		double tmp = a[idx];
		a[idx] = a[l - idx - 1];
		a[l - idx - 1] = tmp;
		idx += offset;
	}
}

int main() {

	int i, n;
	scanf("%i", &n);
	double* arr, * arr_dev, * arr_res;

	arr = (double*)malloc(sizeof(double) * n);
	arr_res = (double*)malloc(sizeof(double) * n);

	for (i = 0; i < n; i++) {
		scanf("%lf", &arr[i]);
	}
	/*
	printf("Vector before inversion\n");
	for (i = 0; i < n; i++) {
		printf("%e ", arr[i]);
	}
	printf("\n");
	*/

	CSC(cudaMalloc(&arr_dev, sizeof(double) * n));

	CSC(cudaMemcpy(arr_dev, arr, sizeof(double) * n, cudaMemcpyHostToDevice));

	reverse_vector<<<2, 32>>>(arr_dev, n);

	CSC(cudaDeviceSynchronize());
	CSC(cudaGetLastError());

	CSC(cudaMemcpy(arr_res, arr_dev, sizeof(double) * n, cudaMemcpyDeviceToHost));

	/*
	printf("Vector after inversion\n");
	for (i = 0; i < n; i++) {
		printf("%e ", arr_res[i]);
	}
	printf("\n");
	*/

	for (i = 0; i < n; i++) {
		if (i < n - 1){
			fprintf(stdout, "%e ", arr_res[i]);
		} else {
			fprintf(stdout, "%e", arr_res[i]);
		}
		
	}


	free(arr);
	free(arr_res);
	CSC(cudaFree(arr_dev));

	return 0;
}