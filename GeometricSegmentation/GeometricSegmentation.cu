#include "GeometricSegmentation.h"


__device__ bool try_read_vertex(int x, int y, const float3* vertexMap, int width, int height, float3& result)
{
	if (x >= 0 && x < width && y >= 0 && y < height)
	{
		float3 vertex = vertexMap[y * width + x];
		if (vertex.x * vertex.x + vertex.y * vertex.y + vertex.z * vertex.z > 0)
		{
			result = vertex;
			return true;
		}
		return false;
	}
	return false;
}

__device__ bool try_read_normal(int x, int y, const float3* normalMap, int width, int height, float3& result)
{
	if (x >= 0 && x < width && y >= 0 && y < height)
	{
		float3 normal = normalMap[y * width + x];
		if (normal.x * normal.x + normal.y * normal.y + normal.z * normal.z > 0)
		{
			result = normal;
			return true;
		}
		return false;
	}
	return false;
}

__device__ void viewpoint_correction(const float3 v, float3& n)
{
	if (n.x * v.x + n.y * v.y + n.z * v.z > 0.0)
	{
		n.x *= -1.0;
		n.y *= -1.0;
		n.z *= -1.0;
	}

}


__global__ void calculate_edge(
	int width, int height,
	float3* vertexMap,
	float3* normalMap,
	unsigned char* segm_bin
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	//printf("%d, %d\n", x, y);
	float3 vc;
	float3 nc;
	float3 v[8];
	float3 n[8];


	int _step = 1;
	int edge = 15;
	if (x < edge || x > width - edge || y < edge || y > height - edge)
	{
		segm_bin[y*width + x] = 0;
		return;
	}

	bool isRead = true;
	isRead = isRead && try_read_vertex(x, y, vertexMap, width, height, vc);
	isRead = isRead && try_read_normal(x, y, normalMap, width, height, nc);
	isRead = isRead && try_read_vertex(x - _step, y - _step, vertexMap, width, height, v[0]);
	isRead = isRead && try_read_normal(x - _step, y - _step, normalMap, width, height, n[0]);
	isRead = isRead && try_read_vertex(x, y - _step, vertexMap, width, height, v[1]);
	isRead = isRead && try_read_normal(x, y - _step, normalMap, width, height, n[1]);
	isRead = isRead && try_read_vertex(x + _step, y - _step, vertexMap, width, height, v[2]);
	isRead = isRead && try_read_normal(x + _step, y - _step, normalMap, width, height, n[2]);
	isRead = isRead && try_read_vertex(x - _step, y + _step, vertexMap, width, height, v[3]);
	isRead = isRead && try_read_normal(x - _step, y + _step, normalMap, width, height, n[3]);
	isRead = isRead && try_read_vertex(x - _step, y, vertexMap, width, height, v[4]);
	isRead = isRead && try_read_normal(x - _step, y, normalMap, width, height, n[4]);
	isRead = isRead && try_read_vertex(x + _step, y + _step, vertexMap, width, height, v[5]);
	isRead = isRead && try_read_normal(x + _step, y + _step, normalMap, width, height, n[5]);
	isRead = isRead && try_read_vertex(x, y + _step, vertexMap, width, height, v[6]);
	isRead = isRead && try_read_normal(x, y + _step, normalMap, width, height, n[6]);
	isRead = isRead && try_read_vertex(x + _step, y, vertexMap, width, height, v[7]);
	isRead = isRead && try_read_normal(x + _step, y, normalMap, width, height, n[7]);

	if (isRead)
	{

		float3 vtmp;
		float3 vtmp_norm;

		float phi = 10000.0;
		float phi_tmp;
		float kai = -1.0;
		float kai_tmp, kai_tmp_norm;

		viewpoint_correction(vc, nc);

		for (int i = 0; i < 8; i++)
		{
			viewpoint_correction(v[i], n[i]);
			vtmp = make_float3(v[i].x - vc.x, v[i].y - vc.y, v[i].z - vc.z);
			//float _length_vtmp = sqrt(vtmp.x * vtmp.x + vtmp.y * vtmp.y + vtmp.z * vtmp.z);


			//vtmp_norm = make_float3(vtmp.x / _length_vtmp, vtmp.y / _length_vtmp, vtmp.z / _length_vtmp);
			kai_tmp = vtmp.x * nc.x + vtmp.y * nc.y + vtmp.z * nc.z;
			//kai_tmp_norm = vtmp_norm.x * nc.x + vtmp_norm.y * nc.y + vtmp_norm.z * nc.z;
			if (kai_tmp > 0.0)
				phi_tmp = 1.0;
			else {
				phi_tmp = nc.x * n[i].x + nc.y * n[i].y + nc.z * n[i].z;
			}
			if (kai_tmp < 0.) kai_tmp = -1.0 * kai_tmp;
			if (i == 0 || kai_tmp > kai)
				kai = kai_tmp;
			if (i == 0 || phi_tmp < phi)
				phi = phi_tmp;
		}
		
		float _dist = sqrt(vc.x * vc.x + vc.y * vc.y + vc.z * vc.z);
		//printf("%f\n", kai);
		float depthUncertaintyCoef = 0.0000285f;
		float _depth = _dist;
		float depthUncertainty = depthUncertaintyCoef * _depth * _depth * 0.5f;
		if (_depth > 1.0f)
		{
			if (kai > depthUncertainty * 200.0 || phi < 0.94f)
			{
				segm_bin[y*width + x] = 0;
			}
			else
				segm_bin[y*width + x] = 255;
		}
		else
		{
			if (kai > depthUncertainty * 200.0 || phi < 0.97f)
				segm_bin[y*width + x] = 0;
			else
				segm_bin[y*width + x] = 255;
		}

	}
	else
		segm_bin[y*width + x] = 0;

}

void GeometricSegmentation::CalcEdge(float3* vertexMap_device, float3* normalMap_device)
{
	calculate_edge << <dim3(width / 32, height / 24), dim3(32, 24) >> >(width, height, vertexMap_device, normalMap_device, segmMap_Device);
	CUDA_SAFE_CALL(cudaMemcpy(segmMap_Host, segmMap_Device, sizeof(unsigned char)*width*height, cudaMemcpyDeviceToHost));
}

__global__ void calculate_edge_vis(
	int width, int height,
	float3* vertexMap,
	float3* normalMap,
	float* kaiMap,
	float* phiMap,
	unsigned char* segm_bin
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	//printf("%d, %d\n", x, y);
	float3 vc;
	float3 nc;
	float3 v[8];
	float3 n[8];


	int _step = 2;
	int edge = 15;
	if (x < edge || x > width - edge || y < edge || y > height - edge)
	{
		segm_bin[y*width + x] = 0;
		kaiMap[y*width+x] = float(0.0);
		phiMap[y*width+x] = float(0.0);
		return;
	}

	bool isRead = true;
	isRead = isRead && try_read_vertex(x, y, vertexMap, width, height, vc);
	isRead = isRead && try_read_normal(x, y, normalMap, width, height, nc);
	isRead = isRead && try_read_vertex(x - _step, y - _step, vertexMap, width, height, v[0]);
	isRead = isRead && try_read_normal(x - _step, y - _step, normalMap, width, height, n[0]);
	isRead = isRead && try_read_vertex(x, y - _step, vertexMap, width, height, v[1]);
	isRead = isRead && try_read_normal(x, y - _step, normalMap, width, height, n[1]);
	isRead = isRead && try_read_vertex(x + _step, y - _step, vertexMap, width, height, v[2]);
	isRead = isRead && try_read_normal(x + _step, y - _step, normalMap, width, height, n[2]);
	isRead = isRead && try_read_vertex(x - _step, y + _step, vertexMap, width, height, v[3]);
	isRead = isRead && try_read_normal(x - _step, y + _step, normalMap, width, height, n[3]);
	isRead = isRead && try_read_vertex(x - _step, y, vertexMap, width, height, v[4]);
	isRead = isRead && try_read_normal(x - _step, y, normalMap, width, height, n[4]);
	isRead = isRead && try_read_vertex(x + _step, y + _step, vertexMap, width, height, v[5]);
	isRead = isRead && try_read_normal(x + _step, y + _step, normalMap, width, height, n[5]);
	isRead = isRead && try_read_vertex(x, y + _step, vertexMap, width, height, v[6]);
	isRead = isRead && try_read_normal(x, y + _step, normalMap, width, height, n[6]);
	isRead = isRead && try_read_vertex(x + _step, y, vertexMap, width, height, v[7]);
	isRead = isRead && try_read_normal(x + _step, y, normalMap, width, height, n[7]);

	if (isRead)
	{

		float3 vtmp;
		float3 vtmp_norm;

		float phi = 10000.0;
		float phi_tmp;
		float kai = -1.0;
		float kai_tmp, kai_tmp_norm;

		viewpoint_correction(vc, nc);

		for (int i = 0; i < 8; i++)
		{
			viewpoint_correction(v[i], n[i]);
			vtmp = make_float3(v[i].x - vc.x, v[i].y - vc.y, v[i].z - vc.z);
			//float _length_vtmp = sqrt(vtmp.x * vtmp.x + vtmp.y * vtmp.y + vtmp.z * vtmp.z);


			//vtmp_norm = make_float3(vtmp.x / _length_vtmp, vtmp.y / _length_vtmp, vtmp.z / _length_vtmp);
			kai_tmp = vtmp.x * nc.x + vtmp.y * nc.y + vtmp.z * nc.z;
			//kai_tmp_norm = vtmp_norm.x * nc.x + vtmp_norm.y * nc.y + vtmp_norm.z * nc.z;
			if (kai_tmp < 0.0)
				phi_tmp = 1.0;
			else {
				phi_tmp = nc.x * n[i].x + nc.y * n[i].y + nc.z * n[i].z;
			}
			if (kai_tmp < 0.) kai_tmp = -1.0 * kai_tmp;
			if (i == 0 || kai_tmp > kai)
				kai = kai_tmp;
			if (i == 0 || phi_tmp < phi)
				phi = phi_tmp;
		}
		kaiMap[y*width+x] = kai;
		phiMap[y*width+x] = phi;
		float _dist = sqrt(vc.x * vc.x + vc.y * vc.y + vc.z * vc.z);
		//printf("%f\n", kai);
		float depthUncertaintyCoef = 0.0000285f;
		float _depth = _dist;
		float depthUncertainty = depthUncertaintyCoef * _depth * _depth * 0.5f;
		if (_depth > 1.0f)
		{
			if (kai > depthUncertainty * 200.0 || phi < 0.94f)
			{
				segm_bin[y*width + x] = 0;
			}
			else
				segm_bin[y*width + x] = 255;
		}
		else
		{
			if (kai > depthUncertainty * 200.0 || phi < 0.97f)
				segm_bin[y*width + x] = 0;
			else
				segm_bin[y*width + x] = 255;
		}

	}
	else
	{
		kaiMap[y*width+x] = float(0.0);
		phiMap[y*width+x] = float(0.0);
		segm_bin[y*width + x] = 0;
	}
}

void GeometricSegmentation::CalcEdge_vis(float3* vertexMap_device, float3* normalMap_device)
{
	calculate_edge_vis << <dim3(width / 32, height / 24), dim3(32, 24) >> >(width, height, vertexMap_device, normalMap_device,kaiMap_Device, phiMap_Device, segmMap_Device);
	CUDA_SAFE_CALL(cudaMemcpy(segmMap_Host, segmMap_Device, sizeof(unsigned char)*width*height, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(kaiMap_Host, kaiMap_Device, sizeof(float)*width*height, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(phiMap_Host, phiMap_Device, sizeof(float)*width*height, cudaMemcpyDeviceToHost));
}