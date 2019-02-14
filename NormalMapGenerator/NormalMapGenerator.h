/////////////////////////////////////////////////////////////////////////////////
// contents :NormalMap generation from SmoothingAreaMap, IntegralImage, vertexMap
// create 	:2013/07/02
// modefied :  
// writer   :Takuya Ikeda 
// other	:
/////////////////////////////////////////////////////////////////////////////////

#ifndef _NORMALMAPgenerator_H_
#define _NORMALMAPgenerator_H_


#include "../Utils/cuda_header.h"

#define BLOCKDIM 16

class NormalMapGenerator{
public:
	NormalMapGenerator(int w, int h);
	~NormalMapGenerator();
	void generateNormalMap(float3* vertices_device);

	//IO
	float3* getNormalMap(void);
private:
	int width;
	int height;
	float3* normalMap;
	void initMemory();
	void computeNormal(float3* vertices_device);
};
#endif 