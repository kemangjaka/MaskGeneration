/////////////////////////////////////////////////////////////////////////////////
// contents :NormalMap generation from SmoothingAreaMap, IntegralImage, vertexMap
// create 	:2013/03/17
// modefied :
// writer   :Takuya Ikeda 
// other	:GPU part
/////////////////////////////////////////////////////////////////////////////////

#include "NormalMapGenerator.h"

__global__ void CalculateNormal(float3* input, float3* normal, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x == 0 || y == 0 || x == width - 1 || y == height - 1)
	{
		normal[y*width + x].x = 0.0;
		normal[y*width + x].y = 0.0;
		normal[y*width + x].z = 0.0;
		return;
	}

	float3 zeroVec;
	zeroVec.x = 0.0;
	zeroVec.y = 0.0;
	zeroVec.z = 0.0;

	if (input[y*width + x].z <= 0.0)
	{
		normal[y*width + x].x = 0.0;
		normal[y*width + x].y = 0.0;
		normal[y*width + x].z = 0.0;
		return;
	}

	float3 xp1_y, xm1_y, x_yp1, x_ym1;
	float3 diff_x, diff_y;
	xp1_y = input[y*width + x + 1];
	xm1_y = input[y*width + x - 1];
	x_yp1 = input[(y + 1)*width + x];
	x_ym1 = input[(y - 1)*width + x];

	if (xp1_y.z == 0.0 || xm1_y.z == 0.0 || x_yp1.z == 0.0 || x_ym1.z == 0.0)
	{
		normal[y*width + x].x = 0.0;
		normal[y*width + x].y = 0.0;
		normal[y*width + x].z = 0.0;
		return;
	}

	// gradients x and y
	diff_x.x = xp1_y.x - xm1_y.x; diff_x.y = xp1_y.y - xm1_y.y; diff_x.z = xp1_y.z - xm1_y.z;
	diff_y.x = x_yp1.x - x_ym1.x; diff_y.y = x_yp1.y - x_ym1.y; diff_y.z = x_yp1.z - x_ym1.z;

	float3 outNormal;

	// cross product
	outNormal.x = (diff_x.y * diff_y.z - diff_x.z*diff_y.y);
	outNormal.y = (diff_x.z * diff_y.x - diff_x.x*diff_y.z);
	outNormal.z = (diff_x.x * diff_y.y - diff_x.y*diff_y.x);

	if (outNormal.x == 0.0 || outNormal.y == 0.0 || outNormal.z == 0.0)
	{
		normal[y*width + x].x = 0.0;
		normal[y*width + x].y = 0.0;
		normal[y*width + x].z = 0.0;
		return;
	}

	float norm = 1.0f / sqrt(outNormal.x * outNormal.x + outNormal.y * outNormal.y + outNormal.z * outNormal.z);
	outNormal.x *= norm; outNormal.y *= norm; outNormal.z *= norm;

	normal[y*width + x].x = outNormal.x; normal[y*width + x].y = outNormal.y; normal[y*width + x].z = outNormal.z;
}

void NormalMapGenerator::generateNormalMap(float3* vertices_device){
	CalculateNormal << <dim3(width / 32, height / 24), dim3(32, 24) >> >
		(vertices_device, normalMap, width, height);

}
