/////////////////////////////////////////////////////////////////////////////////
// contents :NormalMap generation from SmoothingAreaMap, IntegralImage, vertexMap
// create 	:2013/03/17
// modefied :  
// writer   :Takuya Ikeda
// other	:
/////////////////////////////////////////////////////////////////////////////////

//#include "../header/OpenCV.h"
#include "NormalMapGenerator.h"

NormalMapGenerator::NormalMapGenerator(int w, int h){
	this->width = w;
	this->height = h;

	initMemory();
}

NormalMapGenerator::~NormalMapGenerator(){
	cudaFree(normalMap);
}

void NormalMapGenerator::initMemory(){
	CUDA_SAFE_CALL( cudaMalloc((void**) &normalMap, width * height * sizeof(float3)));	
}

float3* NormalMapGenerator::getNormalMap(void){
	return normalMap;	
}

