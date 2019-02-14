#ifndef _UTILS_
#define _UTILS_


#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <array>
#include <stack>
#include <iomanip>
#include <sstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;


vector<string> readDir(string folder);
cv::Mat visualizeFloatImage(cv::Mat img, bool color, bool rev);
 cv::Mat softmaxImage(cv::Mat img);
#endif