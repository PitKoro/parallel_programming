#include <cmath>
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


texture<uchar4, 2, cudaReadModeElementType> rgb_texture;


__device__ double calculate_grad(int x, int y, int* mask)
{
	uchar4 rgb_format;
	double bright;
	double grad = 0;
	int indexing_arr[3] = {-1,0,1};
	int size = 9;
	
	for(int i = 0; i < size; i++)
	{
		rgb_format = tex2D(rgb_texture, x + indexing_arr[i%3], y - indexing_arr[i/3]);
		bright = 0.299 * rgb_format.x + 0.587 * rgb_format.y + 0.114 * rgb_format.z;
		grad += bright * (double)mask[8 - i];
	}

	return grad;
}

__global__ void kernel(uchar4 *output, int h, int w, int* prewitt_mask_x, int* prewitt_mask_y)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;

	int offset_x = blockDim.x * gridDim.x;
	int offset_y = blockDim.y * gridDim.y;

    uchar4 pixel;	

	for (int x = idx; x < w; x += offset_x)
	{
		for (int y = idy; y < h; y += offset_y)
		{
            pixel = tex2D(rgb_texture, x, y);
			double grad_x = calculate_grad(x, y, prewitt_mask_x);
			double grad_y = calculate_grad(x, y, prewitt_mask_y);
			double total = sqrt(grad_x * grad_x + grad_y * grad_y);

			if(total > UCHAR_MAX)
			{ 
				total = UCHAR_MAX;
			}

			output[y * w + x] = make_uchar4(total, total, total, pixel.w);
		}
	}
}


int main()
{
	int w, h;
	char path_to_input_file[255];
	char path_to_output_file[255];
	scanf("%s",path_to_input_file);
	scanf("%s",path_to_output_file);

	FILE* input_file = fopen(path_to_input_file, "rb");
	fread(&w, sizeof(int), 1, input_file);
	fread(&h, sizeof(int), 1, input_file);
	uchar4* img = (uchar4*)malloc(sizeof(uchar4) * w* h);
	fread(img, sizeof(uchar4), w * h, input_file);
	fclose(input_file);

	int size_mask = 9;
	int prewitt_mask_x[size_mask] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
	int prewitt_mask_y[size_mask] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
	
	cudaArray *dev_img;
	
	// привязка изображения к uchar4
	cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();

	rgb_texture.addressMode[0] = cudaAddressModeClamp;
	rgb_texture.addressMode[1] = cudaAddressModeClamp;
	rgb_texture.channelDesc = channel_desc;
	rgb_texture.filterMode  = cudaFilterModePoint;
	rgb_texture.normalized  = false;

	int *dev_prewitt_mask_x;
	int *dev_prewitt_mask_y;

	CSC(cudaMalloc(&dev_prewitt_mask_x, sizeof(int) *size_mask));
	CSC(cudaMalloc(&dev_prewitt_mask_y, sizeof(int) *size_mask));
	CSC(cudaMemcpy(dev_prewitt_mask_x, prewitt_mask_x, size_mask*sizeof(int), cudaMemcpyHostToDevice));
	CSC(cudaMemcpy(dev_prewitt_mask_y, prewitt_mask_y, size_mask*sizeof(int), cudaMemcpyHostToDevice));
	CSC(cudaMallocArray(&dev_img, &channel_desc, w,h));
	CSC(cudaMemcpyToArray(dev_img, 0, 0, img, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));
	CSC(cudaBindTextureToArray(rgb_texture, dev_img, channel_desc));

	uchar4* dev_output_img;
	CSC(cudaMalloc(&dev_output_img, sizeof(uchar4) * w * h));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

	kernel<<<512, 512>>>(dev_output_img, h, w, dev_prewitt_mask_x, dev_prewitt_mask_y);

	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float t;
    cudaEventElapsedTime(&t, start, stop);
	printf("time: %lf\n\n", t);
	cudaEventDestroy(start);
    cudaEventDestroy(stop);
	
	CSC(cudaGetLastError());
	CSC(cudaMemcpy(img, dev_output_img, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
	CSC(cudaUnbindTexture(rgb_texture));

	FILE* output_file = fopen(path_to_output_file, "wb");

	fwrite(&w, sizeof(int), 1, output_file);
	fwrite(&h, sizeof(int), 1, output_file);
	fwrite(img, sizeof(uchar4), w * h, output_file);
	fclose(output_file);

	CSC(cudaFreeArray(dev_img));
	CSC(cudaFree(dev_output_img));
	CSC(cudaFree(dev_prewitt_mask_x));
	CSC(cudaFree(dev_prewitt_mask_y));
	free(img);

	return 0;
}
