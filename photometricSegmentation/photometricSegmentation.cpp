#include "photometricSegmentation.h"
#include "../lock.h"

photometricSegmentation::photometricSegmentation()
{
    std::cout << "Initialize RCF..." << std::endl;
    Py_SetProgramName((wchar_t*)L"rcf");
    Py_Initialize();
    wchar_t const * argv2[] = { L"rcf.py" };
    PySys_SetArgv(1, const_cast<wchar_t**>(argv2));

    std::cout << " * Loading module..." << std::endl;
    pModule = PyImport_ImportModule("rcf");
    if(pModule == NULL) {
        if(PyErr_Occurred()) {
            std::cout << "Python error indicator is set:" << std::endl;
            PyErr_Print();
        }
        throw std::runtime_error("Could not open RCF module.");
    }
    _import_array();
    // Get function
    pExecute = PyObject_GetAttrString(pModule, "execute");
    if(pExecute == NULL || !PyCallable_Check(pExecute)) {
        if(PyErr_Occurred()) {
            std::cout << "Python error indicator is set:" << std::endl;
            PyErr_Print();
        }
        throw std::runtime_error("Could not load function 'execute' from RCF module.");
    }
    std::cout << "* Initialised RCF with thread id : " << std::this_thread::get_id() << std::endl;

}



PyObject *photometricSegmentation::createArguments(cv::Mat rgbImage){
    assert(rgbImage.channels() == 3);
    npy_intp dims[3] = { rgbImage.rows, rgbImage.cols, 3 };
    return PyArray_SimpleNewFromData(3, dims, NPY_UINT8, rgbImage.data); // TODO Release?
}

photometricSegmentation::~photometricSegmentation()
{
    Py_XDECREF(pModule);
    Py_XDECREF(pExecute);
    Py_Finalize();
}

PyObject *photometricSegmentation::getPyObject(const char* name){
    PyObject* obj = PyObject_GetAttrString(pModule, name);
    if(!obj || obj == Py_None) throw std::runtime_error(std::string("Failed to get python object: ") + name);
    return obj;
}

cv::Mat photometricSegmentation::extractImage()
{
    PyObject* pImage = getPyObject("result");
    PyArrayObject *pImageArray = (PyArrayObject*)(pImage);

    unsigned char* pData = (unsigned char*)PyArray_GETPTR1(pImageArray,0);
    npy_intp h = PyArray_DIM(pImageArray,0);
    npy_intp w = PyArray_DIM(pImageArray,1);
    cv::Mat result;
    cv::Mat(h,w, CV_8UC1, pData).copyTo(result);
    Py_DECREF(pImage);
    return result;
}

void photometricSegmentation::segment(cv::Mat img)
{
    //std::cout << "* execute with thread id : " << std::this_thread::get_id() << std::endl;
    Py_XDECREF(PyObject_CallFunctionObjArgs(pExecute, createArguments(img), NULL));
    segmentedImage = extractImage();
}

cv::Mat photometricSegmentation::getsegmImg()
{
    return segmentedImage;
}



photometricSegmentationMT::photometricSegmentationMT()
{

}

void photometricSegmentationMT::startThread(cv::Mat rgbImage)
{
        std::cout << "Initialize RCF..." << std::endl;
    Py_SetProgramName((wchar_t*)L"rcf");
    Py_Initialize();
    wchar_t const * argv2[] = { L"rcf.py" };
    PySys_SetArgv(1, const_cast<wchar_t**>(argv2));

    std::cout << " * Loading module..." << std::endl;
    PyObject* pModule_multi = PyImport_ImportModule("rcf");
    if(pModule_multi == NULL) {
        if(PyErr_Occurred()) {
            std::cout << "Python error indicator is set:" << std::endl;
            PyErr_Print();
        }
        throw std::runtime_error("Could not open RCF module.");
    }
    _import_array();
    // Get function
    PyObject* pExecute_multi = PyObject_GetAttrString(pModule_multi, "execute");
    if(pExecute_multi == NULL || !PyCallable_Check(pExecute_multi)) {
        if(PyErr_Occurred()) {
            std::cout << "Python error indicator is set:" << std::endl;
            PyErr_Print();
        }
        throw std::runtime_error("Could not load function 'execute' from RCF module.");
    }
    std::cout << "* Initialised RCF with thread id : " << std::this_thread::get_id() << std::endl;

    while(1)
    {
        if(cv::sum(rgbImage) == cv::Scalar(0))
            continue;
        mtx_for_segm.lock();
        std::cout << "* execute with thread id : " << std::this_thread::get_id() << std::endl;
        npy_intp dims[3] = { rgbImage.rows, rgbImage.cols, 3 };
        PyObject * obj_py = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, rgbImage.data);
        Py_XDECREF(PyObject_CallFunctionObjArgs(pExecute_multi, obj_py, NULL));
        PyObject* pImage = PyObject_GetAttrString(pModule_multi, "result");
        PyArrayObject *pImageArray = (PyArrayObject*)(pImage);

        unsigned char* pData = (unsigned char*)PyArray_GETPTR1(pImageArray,0);
        npy_intp h = PyArray_DIM(pImageArray,0);
        npy_intp w = PyArray_DIM(pImageArray,1);
        std::cout << h << "," << w << std::endl;
        cv::Mat result;
        cv::Mat(h,w, CV_8UC1, pData).copyTo(result);
        Py_DECREF(pImage);
        mtx_for_segm.unlock();
    }
}