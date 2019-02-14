#include "RGBDSceneReader.h"


RGBDSceneReader::RGBDSceneReader()
{
	width = WIDTH;
	height = HEIGHT;
	scene_num = SCENE_NUM;
	img_dir = "/home/ryo/Dataset/";
	intrinsic = cv::Mat::ones(cv::Size(3, 3), CV_32F);
	intrinsic.at<float>(0, 2) = 320.0;
	intrinsic.at<float>(1, 2) = 240.0;
	intrinsic.at<float>(0, 0) = 570.3;
	intrinsic.at<float>(1, 1) = 570.3;
	MM_PER_M = 10000.0;

}

RGBDSceneReader::~RGBDSceneReader()
{

}

int RGBDSceneReader::getWidth()
{
	return width;
}

int RGBDSceneReader::getHeight()
{
	return height;
}

cv::Mat RGBDSceneReader::getIntrinsics()
{
	return intrinsic;
}

int RGBDSceneReader::getSceneNum()
{
	return scene_num;
}

int RGBDSceneReader::getFrameNum(int sceneId)
{
	stringstream ss;
	ss << setfill('0') << setw(2) << right << to_string(sceneId);
	cout << ss.str() << endl;
	int frame_num = readDir(img_dir + "scene_" + ss.str() + "/rgbs/").size() - 2;
	return frame_num;
}

cv::Mat RGBDSceneReader::getRGBImg(int sceneId, int frameId)
{
	stringstream ss;
	ss << setfill('0') << setw(2) << right << to_string(sceneId);
	string file_path = img_dir + "scene_" + ss.str() + "/rgbs/";
	stringstream frame_id;
	frame_id << setfill('0') << setw(5) << right << to_string(frameId);
	string color_path = file_path + frame_id.str() + "-color.png";
	cv::Mat rgb = cv::imread(color_path);

	return rgb;
}

cv::Mat RGBDSceneReader::getDepthImg(int sceneId, int frameId)
{
	stringstream ss;
	ss << setfill('0') << setw(2) << right << to_string(sceneId);
	string file_path = img_dir + "scene_" + ss.str() + "/depths/";
	stringstream frame_id;
	frame_id << setfill('0') << setw(5) << right << to_string(frameId);
	string depth_path = file_path + frame_id.str() + "-depth.png";
	cv::Mat depth = cv::imread(depth_path, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
	depth.convertTo(depth, CV_32F);

	return depth;
}

string RGBDSceneReader::getRGBFile(int sceneId, int frameId)
{
	stringstream frame_id;
	frame_id << setfill('0') << setw(5) << right << to_string(frameId);
	string color_path = frame_id.str() + "-color.png";
	return color_path;
}