#include <list>
#include <iostream>
#include <vector>
using namespace std;

#include "target.h"

#include <opencv2/opencv.hpp>

#define DEBUG 1

#define OCL 1

#if OCL
#include <opencv2/ocl/ocl.hpp>
using namespace cv::ocl;
typedef oclMat mat;
#include "vision_ocl.h"
#else
#include <opencv2/gpu/gpu.hpp>
using namespace cv::gpu;
typedef GpuMat mat;
#endif

void hsvThreshold(mat &img, mat &result, double hueMin, double hueMax, double satMin, double satMax, double valMin, double valMax) {
  vector<mat> hsvSplit(3);
  split(img, hsvSplit);
  mat hue = hsvSplit[0];
  mat sat = hsvSplit[1];
  mat val = hsvSplit[2];
  threshold(hue, hue, hueMax, 255.0, cv::THRESH_TOZERO_INV);
  threshold(hue, hue, hueMin, 255.0, cv::THRESH_TOZERO);
  threshold(hue, hue, 1, 255.0, cv::THRESH_BINARY);
  threshold(sat, sat, satMax, 255.0, cv::THRESH_TOZERO_INV);
  threshold(sat, sat, satMin, 255.0, cv::THRESH_TOZERO);
  threshold(sat, sat, 1, 255.0, cv::THRESH_BINARY);
  threshold(val, val, valMax, 255.0, cv::THRESH_TOZERO_INV);
  threshold(val, val, valMin, 255.0, cv::THRESH_TOZERO);
  threshold(val, val, 1, 255.0, cv::THRESH_BINARY);
  bitwise_and(hue, sat, result);
  bitwise_and(result, val, result);
}

list<Target> find(cv::Mat &cpuImg, int hueMin, int hueMax, int satMin, int satMax, int valMin, int valMax) {
  list<Target> lst;
  mat img; // GPU image

  //OCL doesn't support BGR2HSV on gpu cvtColor
#if OCL
  cv::cvtColor(cpuImg, cpuImg, cv::COLOR_BGR2HSV);
  img.upload(cpuImg);
#else
  img.upload(cpuImg);
  cvtColor(img, img, cv::COLOR_BGR2HSV);
#endif

  mat bin(img.size(), CV_8UC1);
  hsvThreshold(img, bin, hueMin, hueMax, satMin, satMax, valMin, valMax);

#if DEBUG
  bin.download(cpuImg);
  cv::imshow("Bin", cpuImg);
  cv::waitKey(0);
#endif


  return lst;
}

int main() {
#if OCL
  initOpenCL();
#endif
  cv::Mat img = cv::imread("/Users/vmagro/Developer/frc/RealFullField/0.jpg", cv::IMREAD_COLOR);
  list<Target> targets = find(img, 67, 127, 71, 255, 135, 255);
//  list<Target> targets = find(img, 0, 255, 0, 255, 0, 255);
  cout << "Found " << targets.size() << " targets" << endl;
  return 0;
}
