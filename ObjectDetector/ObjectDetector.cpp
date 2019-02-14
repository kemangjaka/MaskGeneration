#include "ObjectDetector.h"


vector<std::string> readObjFile(string filename)
{
	vector<string> object_names;
	ifstream ifs(filename);
	string str;
	while (std::getline(ifs, str))
	{
		object_names.push_back(str);
	}
	return object_names;

}

ObjectDetector::ObjectDetector(vector<cv::Vec3b> _colors)
{
	object_names = readObjFile("./yolo/coco.names");
	detector = new Detector("./yolo/yolov3.cfg", "./yolo/yolov3.weights");
	detector->nms = 0.02;
	colors = _colors;
}

ObjectDetector::~ObjectDetector()
{

}

void ObjectDetector::detect(cv::Mat img)
{
	bounding.clear();
	vector<bbox_t> _bounding;
	_bounding = detector->detect(img, 0.2);

	for (int i = 0; i < _bounding.size(); i++)
		if (_bounding[i].prob > 0.5)
			bounding.push_back(_bounding[i]);

}

vector<bbox_t> ObjectDetector::getBB()
{
	return bounding;
}

cv::Mat ObjectDetector::visBB(cv::Mat img)
{
	cv::Mat visImg = img.clone();
	for (int i = 0; i < bounding.size(); i++)
	{
		putText(visImg, object_names[bounding[i].obj_id], cv::Point2i(bounding[i].x, bounding[i].y), cv::FONT_HERSHEY_COMPLEX, 1.0, colors[bounding[i].obj_id], 2);
		rectangle(visImg, cv::Point2i(bounding[i].x, bounding[i].y), cv::Point2i(bounding[i].x + bounding[i].w, bounding[i].y + bounding[i].h), colors[bounding[i].obj_id], 3);
	}

	return visImg;
}