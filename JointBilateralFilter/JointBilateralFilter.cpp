#include "JointBilateralFilter.h"


JointBilateralFilter::JointBilateralFilter(int width, int height):
	Width(width),
	Height(height){
	CUDA_SAFE_CALL(cudaMalloc((void **)&Filtered_Device, sizeof(float)*Width*Height));
	}
JointBilateralFilter::~JointBilateralFilter(){
	cudaFree(Filtered_Device);
	Filtered_Device = 0;
}

float* JointBilateralFilter::getFiltered_Device()const{
	return Filtered_Device;
}
float* JointBilateralFilter::getFiltered_Host()const
{
	float* Filtered_Host;
	cudaMallocHost(&Filtered_Host, sizeof(float)*Width*Height);
	cudaMemcpy(Filtered_Host, Filtered_Device, sizeof(float)*Width*Height, cudaMemcpyDeviceToHost);
	return Filtered_Host;
}