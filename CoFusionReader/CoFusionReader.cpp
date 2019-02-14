#include "CoFusionReader.h"


CoFusionReader::CoFusionReader()
{
	width = WIDTH;
	height = HEIGHT;
	scene_num = SCENE_NUM;
	img_dir = "/home/ryo/Dataset/CoFusion/place-items";
	intrinsic = cv::Mat::ones(cv::Size(3, 3), CV_32F);
	intrinsic.at<float>(0, 2) = 320.0;
	intrinsic.at<float>(1, 2) = 240.0;
	intrinsic.at<float>(0, 0) = 528.0;
	intrinsic.at<float>(1, 1) = 528.0;
	MM_PER_M = 10000.0;

}

CoFusionReader::~CoFusionReader()
{

}

int CoFusionReader::getWidth()
{
	return width;
}

int CoFusionReader::getHeight()
{
	return height;
}

cv::Mat CoFusionReader::getIntrinsics()
{
	return intrinsic;
}

int CoFusionReader::getSceneNum()
{
	return scene_num;
}

int CoFusionReader::getFrameNum(int sceneId)
{
	return 754;
}

cv::Mat CoFusionReader::getRGBImg(int sceneId, int frameId)
{
	stringstream frame_id;
	frame_id << setfill('0') << setw(4) << right << to_string(frameId);
	string color_path = img_dir + "/rgbs/Color" + frame_id.str() + ".jpg";
	std::cout << "read image " << color_path << std::endl;
	cv::Mat rgb = cv::imread(color_path);

	return rgb;
}

string CoFusionReader::getRGBFile(int sceneId, int frameId)
{
	stringstream frame_id;
	frame_id << setfill('0') << setw(4) << right << to_string(frameId);
	string path = "Color" + frame_id.str() + ".jpg";
	return path;
}

cv::Mat CoFusionReader::getDepthImg(int sceneId, int frameId)
{
	stringstream frame_id;
	frame_id << setfill('0') << setw(4) << right << to_string(frameId);
	string depth_path = img_dir + "/depths/Depth" + frame_id.str() + ".png";
	cv::Mat depth = cv::imread(depth_path, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
	depth.convertTo(depth, CV_32F);

	return depth;
}