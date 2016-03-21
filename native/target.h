#ifndef TARGET_H
#define TARGET_H

#include <vector>
#include <opencv2/opencv.hpp>

class Position {
public:
  const double x, y;

  Position(const double x, const double y) : x(x), y(y) { }
};

class Size {
public:
  const double width, height;

  Size(const double width, const double height) : width(width), height(height) { }
};

class Target {
public:
  Position *pos;
  Size *size;
  double distance;
  double elevationAngle;
  double azimuth;

  Target(std::vector<cv::Point> contour, cv::Size imgSize);

  virtual ~Target();

  int imageX, imageY;
};

#endif
