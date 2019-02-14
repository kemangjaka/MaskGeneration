#ifndef JOINT_BILATERALFILTER_H
#define JOINT_BILATERALFILTER_H

#include "../Utils/cuda_header.h"

#define MEAN_SIGMA_L 1.2232f

class JointBilateralFilter{
public:
	JointBilateralFilter(int width, int height);
	~JointBilateralFilter();
	void Process(float* depth_device);
	float*				getFiltered_Device()const;
	float*				getFiltered_Host()const;

private:
	int					Width;
	int					Height;
	float*				Filtered_Device;
	void FilterDepth(const float* depth_device, float* filteredDepth_device);
};
#endif 