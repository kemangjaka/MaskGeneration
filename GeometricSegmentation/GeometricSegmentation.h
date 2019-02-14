#pragma once
#ifndef _GEOMETRIC_SEGMENTATION_
#define _GEOMETRIC_SEGMENTATION_

/// Object map is 0 origin
/// invalid label: -1



//#include <cutil_inline.h>
#include "../Utils/cuda_header.h"
#include <opencv2/opencv.hpp>
#include <yolo_v2_class.hpp>

class GeometricSegmentation {
private:
	int width;
	int height;
	int labelNum;
	unsigned char* segmMap_Device;
	unsigned char* segmMap_Host;

	float* phiMap_Device, *phiMap_Host;
	float* kaiMap_Device, *kaiMap_Host;

	cv::Mat segmImg_Host;
	cv::Mat labelImg_Host;

	std::vector<bbox_t> geoBB;

	std::vector<cv::Vec3b> colors;

public:
	GeometricSegmentation(int _w, int _h, std::vector<cv::Vec3b> colors);
	~GeometricSegmentation();
	void CalcEdge(float3* vertexMap_device, float3* normalMap_device);
	void CalcEdge_vis(float3* vertexMap_device, float3* normalMap_device);
	int labeling(bool fillHole, int* dest);
	unsigned char* getSegmMap_Host();
	cv::Mat getSegmImg();
	cv::Mat getLabelImg();
	cv::Mat getKaiImg();
	cv::Mat getPhiImg();
	cv::Mat getVisualizedLabelImg(int* indexMap);
	std::vector<bbox_t> getGeometricBBox();
	cv::Mat visBB(cv::Mat img, std::vector<bbox_t> bbox);
	cv::Mat visBBClass(cv::Mat img, std::vector<bbox_t> bbox);

};

#endif