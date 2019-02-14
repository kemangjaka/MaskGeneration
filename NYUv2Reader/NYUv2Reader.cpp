#include "NYUv2Reader.h"


NYUv2Reader::NYUv2Reader()
{
	width = WIDTH;
	height = HEIGHT;
	scene_num = SCENE_NUM;
	img_dir = "/home/ryo/Dataset/NYUv2";
	intrinsic = cv::Mat::ones(cv::Size(3, 3), CV_32F);
	intrinsic.at<float>(0, 2) = 285.5824;
	intrinsic.at<float>(1, 2) = 209.7362;
	intrinsic.at<float>(0, 0) = 518.8579;
	intrinsic.at<float>(1, 1) = 519.4696;
	MM_PER_M = 10000.0;

}

NYUv2Reader::~NYUv2Reader()
{

}

int NYUv2Reader::getWidth()
{
	return width;
}

int NYUv2Reader::getHeight()
{
	return height;
}

cv::Mat NYUv2Reader::getIntrinsics()
{
	return intrinsic;
}

int NYUv2Reader::getSceneNum()
{
	return scene_num;
}

int NYUv2Reader::getFrameNum(int sceneId)
{
	return 1449;
}

cv::Mat NYUv2Reader::getRGBImg(int sceneId, int frameId)
{
	string color_path = img_dir + "/rgbs/train_" + to_string(frameId) + ".jpg";
	std::cout << "read image " << color_path << std::endl;
	cv::Mat rgb = cv::imread(color_path);

	return rgb;
}

string NYUv2Reader::getRGBFile(int sceneId, int frameId)
{
	string path = "train_" + to_string(frameId) + ".jpg";
	return path;
}

cv::Mat NYUv2Reader::getDepthImg(int sceneId, int frameId)
{
	string depth_path = img_dir + "/depthImgs/train_" + to_string(frameId) + ".png";
	cv::Mat depth = cv::imread(depth_path, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
	depth.convertTo(depth, CV_32F);

	return depth;
}