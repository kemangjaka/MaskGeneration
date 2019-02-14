#pragma once

#include "../Header.h"
#include "../NormalMapGenerator/NormalMapGenerator.h"
#include "../DimensionConvertor/DimensionConvertor.h"
#include "../JointBilateralFilter/JointBilateralFilter.h"
#include "../GeometricSegmentation/GeometricSegmentation.h"
#include "../NYUv2Reader/NYUv2Reader.h"
#include "../RGBDSceneReader/RGBDSceneReader.h"
#include "../CoFusionReader/CoFusionReader.h"
#include "../DatasetReader/DatasetReader.h"
#include "../ObjectDetector/ObjectDetector.h"
//#include "../photometricSegmentation/photometricSegmentation.h"

#include <cuda.h>


#define CUDA_SAFE_CALL(func) \
do { \
     cudaError_t err = (func); \
     if (err != cudaSuccess) { \
         fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
         exit(err); \
     } \
} while(0)

//#include <opencv2/core_detect.hpp>

#define IoU_THRESH 0.6
using namespace cv;
class MainEngine {
private:
	string dataset_name;
	NormalMapGenerator* normEngine;
	DimensionConvertor* convEngine;
	JointBilateralFilter* filtEngine;
	GeometricSegmentation* segmEngine;
	//photometricSegmentation* photoSegmEngine;
	//photometricSegmentationMT* photoSegmEngineMT;
	ObjectDetector* detectEngine;
	RGBDSceneReader* rgbdEngine;
	NYUv2Reader* nyuv2Engine;
	CoFusionReader* cofEngine;
	DatasetReader* dataEngine;
	int width;
	int height;
	int sceneNum;
	vector<cv::Vec3b> random_colors;
	std::ofstream ofsFps;

	cv::Mat rgb, depth, viz_depth,filtDepth, colorDepth, normalMap, isPlane;
	float* inputDepth_Host;
	float3* points3D_Device;
	float3* normalMap_Device;
	int* geoSegMap_Host;
	int* objSegMap_Host;

	void assignClass2GeoSegBB(vector<bbox_t> detect_bbox, vector<bbox_t>& geoseg_bbox);
	void assignClass2GeoSegMap(vector<bbox_t> geoseg_bbox, int* objSegMap_Host);
	cv::Mat estimateMFE(float3* normalMap, int* objSegmMap_Host);
	void geoSeg();

	cv::Mat shared_rgbImage;
	thread* thsegm;
public:
	MainEngine(int dataset_id);
	~MainEngine();
	void Activate();
};