#ifndef _NYUv2READER_H
#define _NYUv2READER_H

#include <opencv2/opencv.hpp>
#include "../DatasetReader/DatasetReader.h"
#include "../Utils/Utils.h"
#define SCENE_NUM 14
#define WIDTH 640
#define HEIGHT 480
using namespace std;

class NYUv2Reader: public DatasetReader {
private:
	int width;
	int height;
	int scene_num;
	string root_dir;
	string img_dir;
	cv::Mat intrinsic;
	float MM_PER_M;
public:
	NYUv2Reader();
	~NYUv2Reader();
	int getWidth() override;
	int getHeight() override;
	cv::Mat getIntrinsics() override;
	int getSceneNum();
	int getFrameNum(int sceneId) override;
	cv::Mat getRGBImg(int sceneId, int frameId) override;
	cv::Mat getDepthImg(int sceneId, int frameId) override;
	string getRGBFile(int sceneId, int frameId) override;
};



#endif