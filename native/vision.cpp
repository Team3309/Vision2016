#include <list>
#include <iostream>
#include <vector>
#include <algorithm>

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

#define PERSPECTIVE_ROWS 280
#define PERSPECTIVE_COLS 400

vector<cv::Point2f> sortCorners(vector<cv::Point> &corners) {
  if (corners.size() != 4) {
    throw "Wrong number of corners";
  }

  //sort by y, pick low 2 for top and high 2 for bottom
  sort(corners.begin(), corners.end(), [](cv::Point &a, cv::Point &b) {
    return a.y < b.y;
  });
  cv::Point2f tl = corners[0].x < corners[1].x ? corners[0] : corners[1];
  cv::Point2f tr = corners[0].x >= corners[1].x ? corners[0] : corners[1];
  cv::Point2f bl = corners[2].x < corners[3].x ? corners[2] : corners[3];
  cv::Point2f br = corners[2].x >= corners[3].x ? corners[2] : corners[3];
  return vector<cv::Point2f> {tl, tr, bl, br};
}

vector<cv::Point2f> getCorners(const vector<cv::Point> &contour) {
  vector<cv::Point> hull;
  cv::convexHull(contour, hull);
  vector<cv::Point> unsortedCorners;
  cv::approxPolyDP(hull, unsortedCorners, 0.05 * cv::arcLength(hull, true), true);
  return sortCorners(unsortedCorners);
}

vector<cv::Point2f> fixTargetPerspective(const vector<cv::Point> &contour, cv::Size binSize, mat &new_bin) {
  cv::Mat beforeWarpCpu = mat(binSize, CV_8UC1);
  vector<vector<cv::Point> > drawingContours;
  drawingContours.push_back(contour);
  cv::drawContours(beforeWarpCpu, drawingContours, -1, 255, -1);
  mat beforeWarp(beforeWarpCpu);

  vector<cv::Point2f> corners = getCorners(contour);
  // get a perspective transformation so that the target is warped as if it was viewed head on
  cv::Size shape(PERSPECTIVE_COLS, PERSPECTIVE_ROWS);
  vector<cv::Point2f> destCorners = {
      cv::Point2f(0, 0),
      cv::Point2f(shape.width, 0),
      cv::Point2f(0, shape.height),
      cv::Point2f(shape.width, shape.height)
  };
  cv::Mat warp = cv::getPerspectiveTransform(corners, destCorners);
  warpPerspective(beforeWarp, new_bin, warp, shape);

  //get it back in cpu to get contours
  cv::Mat fixedCpu;
  new_bin.download(fixedCpu);

  vector<vector<cv::Point> > contours;
  cv::findContours(fixedCpu, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  //convert to point2f contour
  vector<cv::Point2f> first(contours[0].size());
  for (auto it = contours[0].begin(); it != contours[0].end(); ++it) {
    first.push_back(*it);
  }
  return first;
}

double aspectRatioScore(const vector<cv::Point2f> &contour) {
  cv::RotatedRect rect = cv::minAreaRect(contour);

  double ratioScore = 0.0;

  // check to make sure the size is defined to prevent possible division by 0 error
  if (rect.size.width != 0 && rect.size.height != 0) {
    // the target is 1ft8in wide by 1ft2in high, so ratio of width/height is 20/14
    ratioScore = 100 - abs((rect.size.width / rect.size.height) - (20 / 14));
  }

  return ratioScore;
}

double coverageScore(const vector<cv::Point2f> &contour) {
  double contourArea = cv::contourArea(contour);
  cv::Rect bounding = cv::boundingRect(contour);
  //ideal area is 88/280 (2 14x2 strips + 1 16*2 strip = 88) and 14x20 = 280
  if (contourArea > 0) {
    double diff = (bounding.area() / contourArea) - (88.0 / 280.0);
    return 100 - (diff * 5);
  }
  return 0;
}

double momentScore(const vector<cv::Point2f> &contour) {
  cv::Moments moments = cv::moments(contour);
  vector<double> hu;
  cv::HuMoments(moments, hu);
  // hu[6] should be close to 0
  return 100 - (hu[6] * 100);
}

cv::Mat targetShape() {
  cv::Mat shape(PERSPECTIVE_ROWS, PERSPECTIVE_COLS, CV_8UC1);
  //0.125*cols is roughly the width of the tap
  //left vertical
  cv::rectangle(shape, cv::Point(0, 0), cv::Point(PERSPECTIVE_COLS * 0.125, PERSPECTIVE_ROWS), 255, -1);
  //right vertical
  cv::rectangle(shape, cv::Point(PERSPECTIVE_COLS - (PERSPECTIVE_COLS * 0.125), 0), cv::Point(PERSPECTIVE_COLS, PERSPECTIVE_ROWS), 255, -1);
  //bottom horizontal
  cv::rectangle(shape, cv::Point(0, PERSPECTIVE_ROWS - (PERSPECTIVE_COLS * 0.125)), cv::Point(PERSPECTIVE_COLS, PERSPECTIVE_ROWS), 255, -1);
  return shape;
}

mat shape(targetShape());

double profileScore(const mat &bin) {
  mat diff;
  subtract(shape, bin, diff);
  cv::Scalar mean, stdDev;
  meanStdDev(diff, mean, stdDev);

  return (100 - (mean[0]) * 0.25);
}

//return true if contour passes test
bool filterContour(vector<cv::Point> &contour, cv::Size binSize) {
  cv::Rect boundingRect = cv::boundingRect(contour);
  if (boundingRect.area() < 1000 || cv::contourArea(contour, false) < 1000) {
    return false;
  }

  try {
    mat new_bin;
    vector<cv::Point2f> newContour = fixTargetPerspective(contour, binSize, new_bin);

    double aspectRatio = aspectRatioScore(newContour);
    double coverage = coverageScore(newContour);
    double profile = profileScore(new_bin);
    double moment = momentScore(newContour);

    double avg = (aspectRatio + coverage + profile + moment) / 4;

    return avg > 90;
  } catch (...) {
    return false;
  }
}

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
  list<Target> targets;
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

  //erode to remove bad dots
  cv::Mat erosion = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(1, 1));
  erode(bin, bin, erosion);
  //dilate to fill holes
  cv::Mat dilation = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5), cv::Point(2, 2));
  dilate(bin, bin, dilation);

  //now we need to copy back to the CPU to find contours
  bin.download(cpuImg);

  vector<vector<cv::Point> > contours;
  cv::findContours(cpuImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  for (auto it = contours.begin(); it != contours.end(); ++it) {
    if (filterContour(*it, bin.size())) {
      targets.push_back(Target(*it, bin.size()));
    }
  }


  return targets;
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
