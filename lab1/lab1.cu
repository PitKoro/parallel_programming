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
__global__ void kernel(int* first_vector, int* second_vector, int* result_vector, int vector_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    while(idx < vector_size) {
        result_vector[idx] = first_vector[idx] * second_vector[idx];
        idx += offset;
    }
}

// Возвращает кол-во чисел в файле
int get_numbers_count(FILE* input) {
    fseek(input, 0, SEEK_SET);
    int counter = 0;

    while (true) {
    int value;
    if (fscanf(input, "%d", &value) == 1)
        counter++;
    if (feof(input))
        break;
    }

    return counter;
}

// Записывает в массив numbers все числа из файла input
void get_all_numbers(FILE* input, int numbers_count, int* numbers) {
    fseek(input, 0, SEEK_SET);
    for (int i = 0; i < numbers_count; ++i) {
        fscanf(input, "%d", &numbers[i]);
    }
}

// Заполняем массивы векторов
void set_vectors(int *all_numbers, int *first_vector, int *second_vector) {
    int vector_size = all_numbers[0];
    for (int i = 0; i < vector_size; i++) {
        first_vector[i] = all_numbers[i + 1];// +1 потому что первый эл-т в all_numbers - размерность векторов
        second_vector[i] = all_numbers[i + 1 + vector_size];
    }
}

int main() {
    FILE *input_file = fopen("input.txt", "r");
    int numbers_count = get_numbers_count(input_file);

    int *all_numbers = new int[numbers_count];
    get_all_numbers(input_file, numbers_count, all_numbers);
    fclose(input_file);

    int vector_size = all_numbers[0];

    int *first_vector = new int[vector_size];
    int *second_vector = new int[vector_size];

    set_vectors(all_numbers, first_vector, second_vector);

    int *result = new int[vector_size];
    int *device_first_vector = new int[vector_size];
    int *device_second_vector = new int[vector_size];
    int *device_result_vector = new int[vector_size];

    CSC(cudaMalloc(&device_first_vector, sizeof(int) * vector_size));
    CSC(cudaMalloc(&device_second_vector, sizeof(int) * vector_size));
    CSC(cudaMalloc(&device_result_vector, sizeof(int) * vector_size));

    CSC(cudaMemcpy(device_first_vector, first_vector, sizeof(int) * vector_size, cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(device_second_vector, second_vector, sizeof(int) * vector_size, cudaMemcpyHostToDevice));

    kernel<<<256, 256>>>(device_first_vector, device_second_vector, device_result_vector, vector_size);

    CSC(cudaDeviceSynchronize());
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(result, device_result_vector, sizeof(int) * vector_size, cudaMemcpyDeviceToHost));

    CSC(cudaFree(device_first_vector));
    CSC(cudaFree(device_second_vector));
    CSC(cudaFree(device_result_vector));

    // Создание файла для записи результата
    FILE *output_file = fopen("output.txt", "w");

    for (int i = 0; i < vector_size; i++) {
        if (i != vector_size - 1) {
            fprintf(output_file, "%.10e ", double(result[i]));
        } else {
            fprintf(output_file, "%.10e", double(result[i]));
        }
        
    }

    fclose(output_file);

    return 0;

}