#include "GeometricSegmentation.h"

inline bool isIn(int w, int h, int x, int y)
{
	return 0 <= x && x < w && 0 <= y && y < h;
}

inline int compress(std::vector<int>& parents, int a)
{
	while (a != parents[a])
	{
		parents[a] = parents[parents[a]];
		a = parents[a];
	}
	return a;
}

inline int link(std::vector<int>& parents, int a, int b)
{
	a = compress(parents, a);
	b = compress(parents, b);
	if (a < b)
		return parents[b] = a;
	else
		return parents[a] = b;
}

inline int relabel(std::vector<int>& parents)
{
	int index = 0;
	for (int k = 0; k < (int)parents.size(); k++)
	{

		if (k == parents[k])
			parents[k] = index++;
		else
			parents[k] = parents[parents[k]];
	}


	return index;
}

GeometricSegmentation::GeometricSegmentation(int _w, int _h, std::vector<cv::Vec3b> _colors)
{
	width = _w;
	height = _h;

	CUDA_SAFE_CALL(cudaMalloc((void **)&segmMap_Device, sizeof(unsigned char)*width*height));
	CUDA_SAFE_CALL(cudaMallocHost((void **)&segmMap_Host, sizeof(unsigned char)*width*height));

	CUDA_SAFE_CALL(cudaMalloc((void **)&kaiMap_Device,      sizeof(float)*width*height));
	CUDA_SAFE_CALL(cudaMallocHost((void **)&kaiMap_Host,    sizeof(float)*width*height));
	CUDA_SAFE_CALL(cudaMalloc((void **)&phiMap_Device,      sizeof(float)*width*height));
	CUDA_SAFE_CALL(cudaMallocHost((void **)&phiMap_Host,    sizeof(float)*width*height));

	labelImg_Host = cv::Mat::zeros(height, width, CV_32S);
	colors = _colors;
}

GeometricSegmentation::~GeometricSegmentation()
{

	segmMap_Host = 0;
	delete[] segmMap_Host;
	cudaFree(segmMap_Device);
	segmMap_Device = 0;
}
unsigned char* GeometricSegmentation::getSegmMap_Host() {
	cudaMemcpy(segmMap_Host, segmMap_Device, sizeof(unsigned char)*width*height, cudaMemcpyDeviceToHost);

	return segmMap_Host;
}

int GeometricSegmentation::labeling(bool fillHole, int* dest)
{
	
	std::vector<int> parents;
	parents.reserve(5000000);
	int index = 0;
	bool c;
	bool flagA;
	bool flagB;
	for (int j = 0; j < height ; j++)
		for (int i = 0; i < width ; i++)
		{
			c = (segmMap_Host[j*width + i] == 255);
			flagA = (isIn(width, height, i - 1, j) && c == (segmMap_Host[j*width + i - 1] == 255));
			flagB = (isIn(width, height, i, j - 1) && c == (segmMap_Host[(j - 1) *width + i] == 255));

			dest[j * width + i] = index; //coloredLabelImg.at<cv::Vec3b>(j, i) = colors[index];
			if ((flagA | flagB) == true) //if((flagA|flagB|flagC|flagD)==true)
			{
				parents.push_back(index);
				if (flagA) {
					int _l = link(parents, dest[j * width + i], dest[j * width + i - 1]);
					dest[j * width + i] = _l;
				}
				if (flagB)
				{
					int _l = link(parents, dest[j * width + i], dest[(j - 1) * width + i]);
					dest[j * width + i] = _l;
				}
				parents.pop_back();
			}
			else
				parents.push_back(index++);
		}

	if (fillHole) {
		int regions = relabel(parents);
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int j = 0; j<height * width; j++) {
			dest[j] = parents[dest[j]];
		}

		//int* label_count = new int[regions];
		int* label_count = (int *)malloc(sizeof(int) * regions);
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int j = 0; j < regions; j++) 
			label_count[j] = 0;
		for (int j = 0; j<height * width; j++) {
			label_count[dest[j]] += 1;
		}
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int j = 0; j<height * width; j++) {
			//if (source[j * W + i]) {
			if (label_count[dest[j]] < 100) {
				if (segmMap_Host[j] == 255) {
					segmMap_Host[j] = 0;
				}
				else {
					segmMap_Host[j] = 255;
				}

			}
		}
		free(label_count);

		return regions;
	}
	else {
		int regions = relabel(parents);


		std::vector<std::vector<unsigned int>> xs(regions);
		std::vector<std::vector<unsigned int>> ys(regions);

		for (int j = 0; j<height *width; j++) {
			if (segmMap_Host[j] == 255) {
				int label = parents[dest[j]];
				dest[j] = label;
				xs[label].push_back(j % width);
				ys[label].push_back(j / width);
			}
			else {
				dest[j] = -1;
			}
		}
		geoBB.clear();
		geoBB.resize(regions);
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int j = 0; j < regions; j++)
		{
			std::vector<unsigned int> xx = xs[j];
			if (xx.size() == 0)
				continue;
			std::vector<unsigned int> yy = ys[j];

			unsigned int _minx = *std::min_element(xx.begin(), xx.end());
			unsigned int _maxx = *std::max_element(xx.begin(), xx.end());
			geoBB[j].x = _minx;
			geoBB[j].w = _maxx - _minx;

			unsigned int _miny = *std::min_element(yy.begin(), yy.end());
			unsigned int _maxy = *std::max_element(yy.begin(), yy.end());

			geoBB[j].y = _miny;
			geoBB[j].h = _maxy - _miny;
			geoBB[j].obj_id = -1;
		}

		return regions;
	}

}

cv::Mat GeometricSegmentation::visBB(cv::Mat img, std::vector<bbox_t> bbox)
{
	cv::Mat visImg = img.clone();
	for (int i = 0; i < bbox.size(); i++)
		rectangle(visImg, cv::Point2i(bbox[i].x, bbox[i].y), cv::Point2i(bbox[i].x + bbox[i].w, bbox[i].y + bbox[i].h), colors[i], 3);

	return visImg;
}

cv::Mat GeometricSegmentation::visBBClass(cv::Mat img, std::vector<bbox_t> bbox)
{
	cv::Mat visImg = img.clone();
	for (int i = 0; i < bbox.size(); i++)
		if(bbox[i].obj_id != -1)
			rectangle(visImg, cv::Point2i(bbox[i].x, bbox[i].y), cv::Point2i(bbox[i].x + bbox[i].w, bbox[i].y + bbox[i].h), colors[bbox[i].obj_id], 3);

	return visImg;
}

cv::Mat GeometricSegmentation::getVisualizedLabelImg( int* indexMap)
{
	cv::Mat img = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int j = 0; j < width*height; j++)
		if (indexMap[j] != -1)
		{
			img.at<cv::Vec3b>(j / width, j % width) = colors[indexMap[j]];
		}
	return img;
}


cv::Mat GeometricSegmentation::getSegmImg()
{
	segmImg_Host = cv::Mat(cv::Size(width, height), CV_8U, segmMap_Host);
	return segmImg_Host;
}

cv::Mat GeometricSegmentation::getLabelImg()
{
	return labelImg_Host;
}

std::vector<bbox_t> GeometricSegmentation::getGeometricBBox()
{
	return geoBB;
}

cv::Mat GeometricSegmentation::getKaiImg()
{
	cv::Mat kaiImg = cv::Mat(cv::Size(width, height), CV_32F, kaiMap_Host);
	return kaiImg;
}

cv::Mat GeometricSegmentation::getPhiImg()
{
	cv::Mat phiImg = cv::Mat(cv::Size(width, height), CV_32F, phiMap_Host);
	return phiImg;
}