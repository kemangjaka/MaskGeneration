#pragma once


#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>


class photometricSegmentation{
public:
    photometricSegmentation();
    ~photometricSegmentation();
    void segment(cv::Mat img);
    cv::Mat getsegmImg();

private:
    PyObject *pModule;
    PyObject *pExecute;
    PyObject* createArguments(cv::Mat rgbImage);
    cv::Mat extractImage();

  // Warning, getPyObject requiers a decref:
  inline PyObject* getPyObject(const char* name);
    cv::Mat segmentedImage;

};

class photometricSegmentationMT{
public:
    photometricSegmentationMT();
    void startThread(cv::Mat rgbImage);
};