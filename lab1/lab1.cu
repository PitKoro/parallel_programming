#include <stdio.h>
#include <stdlib.h>
#include <string.h>


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
__global__ void kernel(int* first_vector, int* second_vector, int* result_vector, int vector_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    while(idx < vector_size) {
        result_vector[idx] = first_vector[idx] * second_vector[idx];
        idx += offset;
    }
}


int main() {
    int vector_size;
    scanf("%d", &vector_size);

    int first_vector[vector_size];
    int second_vector[vector_size];

    for (int i = 0; i < vector_size; i++) {
        scanf("%d", &first_vector[i]);
    }

    for (int i = 0; i < vector_size; i++) {
        scanf("%d", &second_vector[i]);
    }


    int *result = new int[vector_size];
    int *device_first_vector = new int[vector_size];
    int *device_second_vector = new int[vector_size];
    int *device_result_vector = new int[vector_size];

    CSC(cudaMalloc(&device_first_vector, sizeof(int) * vector_size));
    CSC(cudaMalloc(&device_second_vector, sizeof(int) * vector_size));
    CSC(cudaMalloc(&device_result_vector, sizeof(int) * vector_size));

    CSC(cudaMemcpy(device_first_vector, &first_vector, sizeof(int) * vector_size, cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(device_second_vector, &second_vector, sizeof(int) * vector_size, cudaMemcpyHostToDevice));

    kernel<<<256, 256>>>(device_first_vector, device_second_vector, device_result_vector, vector_size);

    CSC(cudaDeviceSynchronize());
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(result, device_result_vector, sizeof(int) * vector_size, cudaMemcpyDeviceToHost));

    CSC(cudaFree(device_first_vector));
    CSC(cudaFree(device_second_vector));
    CSC(cudaFree(device_result_vector));


    // Вывод в stdout
    for (int i = 0; i < vector_size; i++) {
        if (i != vector_size - 1) {
            fprintf(stdout, "%.10e ", double(result[i]));
        } else {
            fprintf(stdout, "%.10e\n", double(result[i]));
        }
        
    }

    return 0;

}