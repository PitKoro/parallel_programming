#include <stdio.h>
#include <stdlib.h>


#define CSC(call) 														\
	do	{																\
		cudaError_t status = call;										\
		if (status != cudaSuccess) {									\
			fprintf(stderr, "ERROR in %s:%d. Message: %s\n", __FILE__, __LINE__,		\
								cudaGetErrorString(status));			\
			exit(0);													\
		}																\
	} while(0)



// Поэлементно перемножает вектора
__global__ void kernel(double* first_vector, double* second_vector, double* result_vector, double vector_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    if (idx < (int)vector_size) {
        result_vector[idx] = first_vector[idx] * second_vector[idx];
        idx += offset;
    }
}


int main() {
    double vector_size;
    scanf("%lf", &vector_size);

    double first_vector[(int)vector_size];
    double second_vector[(int)vector_size];

    for (int i = 0; i < (int)vector_size; i++) {
        scanf("%lf", &first_vector[i]);
    }

    for (int i = 0; i < (int)vector_size; i++) {
        scanf("%lf", &second_vector[i]);
    }


    double *result = new double[(int)vector_size];
    double *device_first_vector = new double[(int)vector_size];
    double *device_second_vector = new double[(int)vector_size];
    double *device_result_vector = new double[(int)vector_size];

    CSC(cudaMalloc(&device_first_vector, sizeof(double) * (int)vector_size));
    CSC(cudaMalloc(&device_second_vector, sizeof(double) * (int)vector_size));
    CSC(cudaMalloc(&device_result_vector, sizeof(double) * (int)vector_size));

    CSC(cudaMemcpy(device_first_vector, &first_vector, sizeof(double) * (int)vector_size, cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(device_second_vector, &second_vector, sizeof(double) * (int)vector_size, cudaMemcpyHostToDevice));

    kernel<<<256, 256>>>(device_first_vector, device_second_vector, device_result_vector, (int)vector_size);

    CSC(cudaDeviceSynchronize());
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(result, device_result_vector, sizeof(double) * (int)vector_size, cudaMemcpyDeviceToHost));

    CSC(cudaFree(device_first_vector));
    CSC(cudaFree(device_second_vector));
    CSC(cudaFree(device_result_vector));


    // Вывод в stdout
    for (int i = 0; i < (int)vector_size; i++) {
        if (i != (int)vector_size - 1) {
            fprintf(stdout, "%.10e ", result[i]);
        } else {
            fprintf(stdout, "%.10e", result[i]);
        }
        
    }

    return 0;

}