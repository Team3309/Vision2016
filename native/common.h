//
// Created by Vinnie Magro on 3/21/16.
//

#ifndef COMMON_HPP
#define COMMON_HPP

#include <vector>
using namespace std;
#include <opencv2/opencv.hpp>

vector <cv::Point2f> sortCorners(vector <cv::Point> &corners);
vector <cv::Point2f> getCorners(const vector <cv::Point> &contour);

#endif
