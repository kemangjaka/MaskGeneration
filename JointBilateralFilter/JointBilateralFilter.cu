#include "JointBilateralFilter.h"

__global__ void filterDepth_device(const float* inputDepth, float* outputDepth, int width, int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < 2 || x > width - 2 || y < 2 || y > height - 2)
		outputDepth[x + y*width] = 0.0;
	float z, tmpz, dz, final_depth = 0.0f, w, w_sum = 0.0f;

	z = inputDepth[x + y * width];
	if (z == 0.0f) { outputDepth[x + y * width] = 0.0f; return; }

	float sigma_z = 1.0f / (0.0012f + 0.0019f*(z - 0.4f)*(z - 0.4f) + 0.0001f / sqrt(z) * 0.25f);

	for (int i = -2, count = 0; i <= 2; i++)
		for (int j = -2; j <= 2; j++, count++)
	{
		tmpz = inputDepth[(x + j) + (y + i) * width];
		if (tmpz == 0.0f)
			continue;
		dz = (tmpz - z);
		dz *= dz;
		w = exp(-0.5f * ((abs(i) + abs(j))*MEAN_SIGMA_L*MEAN_SIGMA_L + dz * sigma_z * sigma_z));
		w_sum += w;
		final_depth += w*tmpz;
	}

	final_depth /= w_sum;
	outputDepth[x + y*width] = final_depth;
}


void JointBilateralFilter::FilterDepth(const float* depth_device, float* filteredDepth_device)
{
	dim3 blockSize(32, 24);
	dim3 gridSize((int)ceil((float)Width / (float)blockSize.x), (int)ceil((float)Height / (float)blockSize.y));

	filterDepth_device << <gridSize, blockSize >> >(depth_device, filteredDepth_device, Width, Height);
}

void JointBilateralFilter::Process(float* depth_host) {
	float* depth_device;
	CUDA_SAFE_CALL( cudaMalloc((void **)&depth_device, sizeof(float)*Width*Height));
	CUDA_SAFE_CALL( cudaMemcpy(depth_device, depth_host, sizeof(float)*Width*Height, cudaMemcpyHostToDevice));
	FilterDepth(depth_device, Filtered_Device);
	FilterDepth(Filtered_Device, depth_device);
	FilterDepth(depth_device, Filtered_Device);
	FilterDepth(Filtered_Device, depth_device);
	FilterDepth(depth_device, Filtered_Device);
	cudaFree(depth_device);
	depth_device = 0;
}
