#include "PosEstimation.h"
#include <iostream>
#include <chrono>

PosEstimation::PosEstimation(std::string path)
{
    posePath = path;
    cv::FileStorage fs;
    fs.open(posePath, cv::FileStorage::READ);
    fs["A"] >> aMatrix;
    fs["R"] >> rMatrix;
    fs["t"] >> tMatrix;
    fs["P"] >> pMatrix;
    camPosition = calCamPosition(rMatrix, tMatrix);
}

PosEstimation::~PosEstimation()
{
}

cv::Point3d PosEstimation::getCamPosition(){
    return camPosition;
}

cv::Point3d PosEstimation::convertFrom2DS(cv::Point2d uv, float s){
    cv::Mat suv = s * (cv::Mat_<double>(3, 1) << uv.x, uv.y, 1);

    // Point in world coordinates
    cv::Mat X_c = aMatrix.inv() * suv; // 3x1

    cv::Mat XYZ = rMatrix.inv() * (X_c - tMatrix); // 3x1
    return cv::Point3d(XYZ.at<double>(0, 0), XYZ.at<double>(0, 1), XYZ.at<double>(0, 2));
}
#if 0
cv::Point3d PosEstimation::convertFrom2D(cv::Point2d uv, int radius)
{
    cv::Mat A_formalized = aMatrix.clone();
    A_formalized.at<double>(0, 2) = 0;
    A_formalized.at<double>(1, 2) = 0;
    //    cout<<"A_formalized"<<endl<<A_formalized<<endl;

    cv::Mat deltaXYZ = (cv::Mat_<double>(3, 1) << 40, 40, 0);
    cv::Mat sXdeltaUV = A_formalized * deltaXYZ;
    double s = sXdeltaUV.at<double>(0, 0) / radius / 2;
    //    cout<<"sXdeltaUV:"<<endl<<sXdeltaUV<<endl;
    //    cout<<"s:"<<s<<endl;

    cv::Mat suv = s * (cv::Mat_<double>(3, 1) << uv.x, uv.y, 1);

    // Point in world coordinates
    cv::Mat X_c = aMatrix.inv() * suv; // 3x1

    cv::Mat XYZ = rMatrix.inv() * (X_c - tMatrix); // 3x1
    return cv::Point3d(XYZ.at<double>(0, 0), XYZ.at<double>(0, 1), XYZ.at<double>(0, 2));
}
#else
cv::Point3d PosEstimation::convertFrom2D(cv::Point2d uv, int radius)
{
    cv::Mat A_formalized = aMatrix.clone();
    A_formalized.at<double>(0, 2) = 0;
    A_formalized.at<double>(1, 2) = 0;
    //    cout<<"A_formalized"<<endl<<A_formalized<<endl;

    cv::Mat deltaXYZ = (cv::Mat_<double>(3, 1) << 40, 40, 0);
    cv::Mat sXdeltaUV = A_formalized * deltaXYZ;
    double s = sXdeltaUV.at<double>(0, 0) / radius / 2;

    cv::Mat suv = s * (cv::Mat_<double>(3, 1) << uv.x, uv.y, 1);
    
    // Point in world coordinates
    cv::Mat X_c = aMatrix.inv() * suv; // 3x1

    cv::Mat XYZ = rMatrix.inv() * (X_c - tMatrix); // 3x1
    return cv::Point3d(XYZ.at<double>(0, 0), XYZ.at<double>(0, 1), XYZ.at<double>(0, 2));
}

#endif