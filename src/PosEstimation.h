#ifndef _POSESTIMATION_
#define _POSESTIMATION_
#pragma once

#include <string>
#include <opencv2/core.hpp>
#include "geometryCalculation.hpp"

class PosEstimation
{
private:
    std::string posePath = "";
    cv::Mat aMatrix, rMatrix, tMatrix, pMatrix;
    cv::Point3d camPosition;
    double s;
public:
    PosEstimation(std::string path);
    ~PosEstimation();
    cv::Point3d getCamPosition();
    cv::Point3d convertFrom2D(cv::Point2d uv, int radius);
    cv::Point3d convertFrom2DS(cv::Point2d uv, float s);
};



#endif