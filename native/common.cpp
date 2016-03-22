#include <vector>
#include <algorithm>
using namespace std;
#include <opencv2/opencv.hpp>

vector <cv::Point2f> sortCorners(vector <cv::Point> &corners) {
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
  return vector < cv::Point2f > {tl, tr, bl, br};
}

vector <cv::Point2f> getCorners(const vector <cv::Point> &contour) {
  vector <cv::Point> hull;
  cv::convexHull(contour, hull);
  vector <cv::Point> unsortedCorners;
  cv::approxPolyDP(hull, unsortedCorners, 0.05 * cv::arcLength(hull, true), true);
  return sortCorners(unsortedCorners);
}
