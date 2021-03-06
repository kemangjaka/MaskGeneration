#include "DimensionConvertor.h"

void DimensionConvertor::setCameraParameters(const cv::Mat_<double> intrinsic, int width, int height){
		//Ε_£focal_length
		Fx = (float)intrinsic.at<double>(0, 0);
		Fy = (float)intrinsic.at<double>(1, 1);
		//ζS
		Cx = (int)intrinsic.at<double>(0, 2);
		Cy = (int)intrinsic.at<double>(1, 2);
	
		this->Width = width;
		this->Height = height;
}

DimensionConvertor::~DimensionConvertor(){
	Fx = 0.0;
	Fy = 0.0;
	Cx = 0;
	Cy = 0;
	Width = 0;
	Height = 0;
}
