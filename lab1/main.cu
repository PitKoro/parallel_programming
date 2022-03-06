#include <iostream>

__global__ void kernel(int* first_vector, int* second_vector, int* result_vector, int vector_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    while(idx < vector_size) {
        result_vector[idx] = first_vector[idx] * second_vector[idx];
        idx += offset;
    }
}

int main() {
    int vector_size = 3;
    int first_vector[3] = {1, 2, 3};
    int second_vector[3] = {4, 5, 6};
    int* result = new int[vector_size];

    int* device_first_vector = new int[vector_size];
    int* device_second_vector = new int[vector_size];
    int* device_result_vector = new int[vector_size];

    cudaMalloc(&device_first_vector, sizeof(int) * vector_size);
    cudaMalloc(&device_second_vector, sizeof(int) * vector_size);
    cudaMalloc(&device_result_vector, sizeof(int) * vector_size);

    cudaMemcpy(device_first_vector, &first_vector, sizeof(int) * vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_second_vector, &second_vector, sizeof(int) * vector_size, cudaMemcpyHostToDevice);

    kernel<<<256, 256>>>(device_first_vector, device_second_vector, device_result_vector, vector_size);

    cudaDeviceSynchronize();
    cudaGetLastError();

    cudaMemcpy(result, device_result_vector, sizeof(int) * vector_size, cudaMemcpyDeviceToHost);

    cudaFree(device_first_vector);
    cudaFree(device_second_vector);
    cudaFree(device_result_vector);

    for (int i = 0; i < vector_size; i++) {
        printf("%d ", result[i]);
    }

    printf("\n");

    return 0;

}