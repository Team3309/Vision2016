#include "target.h"
#include "common.h"
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

#define PI 3.14159265

#define degrees(x) (x * (180 / PI))
#define radians(x) (x * (PI / 180))

using namespace std;

Position *getCenter(vector<cv::Point> contour, cv::Size imgSize) {
  vector<cv::Point2f> corners = getCorners(contour);
  return new Position(((corners[2].x + corners[3].x) / 2) / imgSize.width, ((corners[2].y + corners[3].y) / 2) / imgSize.height);
}

double targetAzimuth(const double x) {
  const double cameraFovHoriz = 53.50;
  //x is [-1,1] position in image
  return (cameraFovHoriz / 2) * x;
}

double targetElevation(const double distInches) {
  double deltaHeight = 96.5 - 13;  // 96.5 inches off the ground - 13 inches height of shooter
  return degrees(atan2(deltaHeight, distInches));
}

double targetDistance(const double y) {
  const double cameraFovVert = 41.41;
  const double cameraPitch = 40;
  // units in inches
  const double targetHeight = 83;
  const double cameraHeight = 9;
  //y is [1,-1] y coord from top-bottom
  return (targetHeight - cameraHeight) / tan(radians((y * cameraFovVert / 2) + cameraPitch));
}

Target::Target(vector<cv::Point> contour, cv::Size imgSize) {
  this->pos = getCenter(contour, imgSize);
  this->imageX = (int) (this->pos->x * imgSize.width);
  this->imageY = (int) (this->pos->y * imgSize.height);

  cv::RotatedRect boundingRect = cv::minAreaRect(contour);
  this->size = new Size(boundingRect.size.width / imgSize.width, boundingRect.size.height / imgSize.height);

  this->distance = targetDistance(this->pos->y);
  this->elevationAngle = targetElevation(this->distance);
  this->azimuth = targetAzimuth(this->pos->x);
}

Target::~Target() {
  delete pos;
  delete size;
}