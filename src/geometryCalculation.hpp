// C++
#include <iostream>
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

cv::Point2f systemEquation2(float a1, float b1, float c1, float a2, float b2, float c2, float defaultValue = 0.5);
cv::Point3f midPoint(cv::Point3f a, cv::Point3f b, float t = 0.5);
float norm(cv::Point3f a, cv::Point3f b);

cv::Point3d convertFrom2D(cv::Point2d uv, int radius, cv::Mat A, cv::Mat R, cv::Mat T);

cv::Point3d convertFrom2D(cv::Point2d uv, cv::Mat A, cv::Mat R, cv::Mat T);

// return distance and closest point of 2 lines: ab and cd
void analyze2Line(cv::Point3f a, cv::Point3f b, cv::Point3f c, cv::Point3f d, float &distance, cv::Point3f &intersertPoint);

cv::Point3d calCamPosition(cv::Mat rMatrix, cv::Mat tMatrix);

cv::Point2d from3dTo2d(cv::Point3d worldPoint, cv::Mat pMatrix, cv::Mat aMatrix);