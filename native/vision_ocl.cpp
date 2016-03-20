#include "vision_ocl.h"
#include <opencv2/opencv.hpp>
#include <opencv2/ocl/ocl.hpp>
using namespace cv::ocl;

void initOpenCL() {
  cv::ocl::DevicesInfo devInfo;
  int res = cv::ocl::getOpenCLDevices(devInfo);
  if(res == 0)
  {
      std::cout << "There is no OPENCL Here !" << std::endl;
  }else
  {
      for(int i = 0 ; i < devInfo.size() ; ++i)
      {
          std::cout << "Device : " << devInfo[i]->deviceName << " is present" << std::endl;
      }
  }
  cv::ocl::setDevice(devInfo[0]);        // select device to use
  std::cout << CV_VERSION_EPOCH << "." << CV_VERSION_MAJOR << "." << CV_VERSION_MINOR << std::endl;
}
