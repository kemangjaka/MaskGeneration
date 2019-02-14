#ifndef _OBJECT_DETECTOR_
#define _OBJECT_DETECTOR_
#define OPENCV
#include <yolo_v2_class.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;


class ObjectDetector
{
private:
	Detector* detector;
	vector<bbox_t> bounding;
	vector<string> object_names;
	std::vector<cv::Vec3b> colors;
public:
	ObjectDetector(vector<cv::Vec3b> colors);
	~ObjectDetector();
	void detect(cv::Mat img);
	vector<bbox_t> getBB();
	cv::Mat visBB(cv::Mat img);
};

#endif