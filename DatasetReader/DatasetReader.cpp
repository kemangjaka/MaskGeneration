#include "./DatasetReader.h"

DatasetReader::DatasetReader()
{
	width = 0;
	height = 0;
	intrinsic = cv::Mat::ones(cv::Size(3, 3), CV_32F);
}

DatasetReader::~DatasetReader()
{

}

int DatasetReader::getWidth()
{
	return width;
}

int DatasetReader::getHeight()
{
	return height;
}

cv::Mat DatasetReader::getIntrinsics()
{
	return intrinsic;
}

int DatasetReader::getFrameNum(int sceneId)
{
	return 0;
}

int DatasetReader::getSceneNum()
{
	return 1;
}

std::string DatasetReader::getRGBFile(int sceneId, int frameId)
{
	return "";
}

cv::Mat DatasetReader::getRGBImg(int sceneId, int frameId)
{
	cv::Mat rgb = cv::Mat::zeros(cv::Size(1, 1), CV_8UC3);
	return rgb;
}

cv::Mat DatasetReader::getDepthImg(int sceneId, int frameId)
{
	cv::Mat depth = cv::Mat::zeros(cv::Size(1, 1), CV_32F);
	return depth;
}