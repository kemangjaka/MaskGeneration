#include "Utils.h"


vector<string> readDir (string dir)
{
	vector<string> fileList;
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error opening " << dir << endl;
		return fileList;
    }

    while ((dirp = readdir(dp)) != NULL) {
        fileList.push_back(string(dirp->d_name));
    }
    closedir(dp);
    return fileList;
}

/*
 vector<string> readDir(string folder) {
	vector<string> fileList;
	HANDLE hFind;
	WIN32_FIND_DATA fd;

	std::stringstream ss;
	ss << folder;
	string::iterator itr = folder.end();
	itr--;
	if (*itr != '\\') ss << '\\';
	ss << "*.*";

	hFind = FindFirstFile(ss.str().c_str(), &fd);

	// �������s
	if (hFind == INVALID_HANDLE_VALUE) {
		exit(1);
	}

	do {
		char *file = fd.cFileName;
		string str = file;
		fileList.push_back(str);
	} while (FindNextFile(hFind, &fd)); //���̃t�@�C����T��

										// hFind�̃N���[�Y
	FindClose(hFind);

	return fileList;
}
*/
 cv::Mat visualizeFloatImage(cv::Mat img, bool color, bool rev)
 {
	double _min, _max;
	cv::minMaxIdx(img, &_min, &_max);
	cv::Mat viz_depth, colorDepth;
	cv::convertScaleAbs(img, viz_depth, 255 / _max);
	if(rev)
		viz_depth = 255 - viz_depth;
	if (color)
	{
		cv::applyColorMap(viz_depth, colorDepth, cv::COLORMAP_JET);
	}else
		cv::cvtColor(viz_depth,colorDepth, CV_GRAY2BGR);
	
	return colorDepth;
 }

 cv::Mat softmaxImage(cv::Mat img)
 {
	float sum = cv::sum(img)[0];
	cout << sum << endl;
	sum = std::exp(sum);
	cv::Mat newMat;
	cv::exp(img, newMat);
	newMat = newMat / sum;
	double _min, _max;
	cv::minMaxIdx(newMat, &_min, &_max);
	cout << _min << "," << _max << endl;
	newMat = 255 * newMat;
	newMat.convertTo(newMat, CV_8U);
	cv::Mat colored;
	cv::cvtColor(newMat,colored, CV_GRAY2BGR);

	return colored;
 }