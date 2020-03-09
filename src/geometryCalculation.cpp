#include "geometryCalculation.hpp"

cv::Point2f systemEquation2(float a1, float b1, float c1, float a2, float b2, float c2, float defaultValue)
{
    float delta = a1 * b2 - a2 * b1;
    float deltaX = c1 * b2 - c2 * b1;
    float deltaY = a1 * c2 - a2 * c1;
    if (delta > -0.00001 && delta < 0.00001)
        return cv::Point2f(defaultValue, defaultValue);
    else
        return cv::Point2f(deltaX * 1.0 / delta, deltaY * 1.0 / delta);
}

cv::Point3f midPoint(cv::Point3f a, cv::Point3f b, float t)
{
    return a * (1 - t) + b * t;
}
float norm(cv::Point3f a, cv::Point3f b)
{
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z));
}

cv::Point3d convertFrom2D(cv::Point2d uv, int radius, cv::Mat A, cv::Mat R, cv::Mat T)
{
    cv::Mat A_formalized = A.clone();
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
    cv::Mat X_c = A.inv() * suv; // 3x1

    cv::Mat XYZ = R.inv() * (X_c - T); // 3x1
    return cv::Point3d(XYZ.at<double>(0, 0), XYZ.at<double>(0, 1), XYZ.at<double>(0, 2));
}

cv::Point3d convertFrom2D(cv::Point2d uv, cv::Mat A, cv::Mat R, cv::Mat T)
{
    cv::Mat suv = 1000 * (cv::Mat_<double>(3, 1) << uv.x, uv.y, 1);

    // Point in world coordinates
    cv::Mat X_c = A.inv() * suv; // 3x1

    cv::Mat XYZ = R.inv() * (X_c - T); // 3x1
    return cv::Point3d(XYZ.at<double>(0, 0), XYZ.at<double>(0, 1), XYZ.at<double>(0, 2));
}

// return distance and closest point of 2 lines: ab and cd
void analyze2Line(cv::Point3f a, cv::Point3f b, cv::Point3f c, cv::Point3f d, float &distance, cv::Point3f &intersertPoint)
{
    cv::Point3f vecAB = b - a;
    cv::Point3f vecCD = d - c;
    cv::Point3f vecAC = c - a;
    float a1 = vecAB.x * vecAB.x + vecAB.y * vecAB.y + vecAB.z * vecAB.z;
    float b1 = -(vecAB.x * vecCD.x + vecAB.y * vecCD.y + vecAB.z * vecCD.z);
    float c1 = vecAB.x * vecAC.x + vecAB.y * vecAC.y + vecAB.z * vecAC.z;
    float a2 = vecCD.x * vecAB.x + vecCD.y * vecAB.y + vecCD.z * vecAB.z;
    float b2 = -(vecCD.x * vecCD.x + vecCD.y * vecCD.y + vecCD.z * vecCD.z);
    float c2 = vecCD.x * vecAC.x + vecCD.y * vecAC.y + vecCD.z * vecAC.z;
    cv::Point2f tu = systemEquation2(a1, b1, c1, a2, b2, c2);
    float t = tu.x;
    float u = tu.y;
    cv::Point3f i = midPoint(a, b, t);
    cv::Point3f j = midPoint(c, d, u);
    distance = norm(i, j);
    intersertPoint = midPoint(i, j);
}

cv::Point3d calCamPosition(cv::Mat rMatrix, cv::Mat tMatrix)
{
    //calculate cam position first point
    cv::Mat rvec;
    cv::Rodrigues(rMatrix, rvec);
    cv::Mat tempR = rvec.t();
    cv::Mat camposeampose = -rMatrix.inv() * tMatrix; // -tempR * tMatrix;
    return cv::Point3f(camposeampose.at<double>(cv::Point(0, 0)),
                       camposeampose.at<double>(cv::Point(1, 0)),
                       camposeampose.at<double>(cv::Point(2, 0)));
}

cv::Point2d from3dTo2d(cv::Point3d worldPoint, cv::Mat pMatrix, cv::Mat aMatrix)
{
    //calculate object position 2d
    cv::Mat worldCoor = (cv::Mat_<double>(4, 1) << worldPoint.x, worldPoint.y, worldPoint.z, 1);
    cv::Mat uv = pMatrix * worldCoor;

    uv = aMatrix * pMatrix * worldCoor;

    // Normalization of [u v]'
    cv::Point2f point2d;
    point2d.x = (float)(uv.at<double>(0) / uv.at<double>(2));
    point2d.y = (float)(uv.at<double>(1) / uv.at<double>(2));
    return point2d;
}