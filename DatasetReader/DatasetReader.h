#ifndef _DATASET_READER_
#define _DATASET_READER_

#include <opencv2/opencv.hpp>
#include <iostream>
class DatasetReader {
private:
	int width;
	int height;
	cv::Mat intrinsic;
public:
	DatasetReader();
	~DatasetReader();
	virtual int getWidth();
	virtual int getHeight();
	virtual cv::Mat getIntrinsics();
	virtual int getFrameNum(int sceneId);
	virtual int getSceneNum();
	virtual cv::Mat getRGBImg(int sceneId, int frameId);
	virtual cv::Mat getDepthImg(int sceneId, int frameId);
	virtual std::string getRGBFile(int sceneId, int frameId);
};


#endif