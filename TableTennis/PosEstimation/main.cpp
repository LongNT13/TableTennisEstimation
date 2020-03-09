// C++
#include <iostream>
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "constants.h"
#include "ball_segmentation.h"
#include "ball_detection.h"
#include "ball_tracking.h"

using namespace cv;
using namespace std;


void drawHoughCircles(Mat& frame, vector<Vec3f> circles);
void drawTrackingInfo(Mat& frame, vector<Point> positions);
void drawTrackingInfo(Mat& frame, vector<Point> positions, Rect roi_rect);

Point3d convertFrom2D(Point2d uv, int radius){
    FileStorage fs;
    fs.open("pose1.yml", FileStorage::READ);
    
    Mat A, R, t;
    fs["A"]>>A;
    fs["R"]>>R;
    fs["t"]>>t;
   cout<<"A"<<A<<endl;
   cout<<"R"<<R<<endl;
   cout<<"t"<<t<<endl;

    Mat A_formalized = A.clone();
    A_formalized.at<double>(0,2) = 0;
    A_formalized.at<double>(1,2) = 0;
//    cout<<"A_formalized"<<endl<<A_formalized<<endl;

    Mat deltaXYZ = (Mat_<double>(3,1) << 40, 40, 0);
    Mat sXdeltaUV = A_formalized*deltaXYZ;
    double s = sXdeltaUV.at<double>(0,0)/radius/2;
//    cout<<"sXdeltaUV:"<<endl<<sXdeltaUV<<endl;
//    cout<<"s:"<<s<<endl;

    Mat suv = s*(Mat_<double>(3,1) << uv.x, uv.y, 1);

    // Point in world coordinates
    cv::Mat X_c = A.inv()*suv ; // 3x1

    cv::Mat XYZ = R.inv() * ( X_c - t ); // 3x1
    return Point3d(XYZ.at<double>(0,0), XYZ.at<double>(0,1),XYZ.at<double>(0,2));
}

int main(int argc, char const *argv[])
{
    Mat frame, roi_blurred, roi_binarized;
    vector<Vec3f> circles;

    // we save info from previous iterations
    Mat frame_previous;  // the previous frame
    vector<Point> positions;  // the history of all detected positions (0 or 1 per frame)
    bool ball_found_prev = false;  // whether we found ball during previous iteration

    //frame = imread("/home/parallels/Code/build-PoseEstimation-Desktop_Qt_5_13_0_GCC_64bit-Debug/Data/table.png", 1 );
    frame = imread("/media/long/Data1/materials/thesisM1/internshipM1/TableTennis/test/Data/cam1.png", 1 );
    bool ball_found = false;
    Point ball_position;

    Mat roi;
    Rect roi_rect(96,466,1335, 541);
    roi = Mat(frame, roi_rect);

    // we blur the picture to remove the noise
    //roi_blurred = roi.clone();
    GaussianBlur(roi, roi_blurred, Size(BLUR_KERNEL_LENGTH, BLUR_KERNEL_LENGTH), 0, 0);

    // ===== first try =====
    // with a threshold segmentation and a Hough transform
    thresholdSegmentation(roi_blurred, roi_binarized);
    detectBallWithHough(roi_binarized, roi_rect, circles);
    HoughCircles( roi_binarized, circles, CV_HOUGH_GRADIENT, 1, 1000, 81, 4, 11, 15 );
    cout<<"No circle"<<circles.size()<<endl;

    // if the number of circles found by the Hough transform is exactly 1,
    // we accept that circle as the correct ball position

    // if (circles.size() == 1) {
    //     ball_found = true;
    //     ball_position = Point(circles[0][0], circles[0][1]);
    //     #ifdef SHOW_WINDOWS
    //         //drawHoughCircles(frame, circles);
    //     cv::circle(frame, ball_position, circles[0][2], cv::Scalar(255,255,0), 2);
    //     Point3d XYZ= convertFrom2D(ball_position, circles[0][2]+1);
    //     string posStr = "(" + std::to_string(int(XYZ.x)) + ", " + std::to_string(int(XYZ.y)) + ", " + std::to_string(int(XYZ.z)) + ")";
    //     cv::putText(frame, posStr,  Point(ball_position.x + 10,ball_position.y + 20),
    //                 FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,200,250), 1, CV_AA);
    //     #endif
    // }

    // if (ball_found == false) {
    //     // ===== second try =====
    //     // we use OpenCV convex hull algorithm to detect shapes
    //     // (the ball is not detected by Hough transform if it is not round enough)
    //     vector<Point> possible_positions;
    //     detectBallWithContours(roi_binarized, roi_rect, possible_positions);

    //     // if the number of positions found by the Hough transform is exactly 1,
    //     // we accept that position as the correct ball position
    //     if (possible_positions.size() == 1) {
    //         ball_found = true;
    //         ball_position = possible_positions[0];
    //     }
    // }
    //Long
    std::cout << __LINE__ << std::endl;
    cv::Point p(300,300);
    ball_found = true;
    cv::circle(frame, p, 30, cv::Scalar(255,255,0), 2);
    std::cout << __LINE__ << std::endl;
    Point3d XYZ= convertFrom2D(p, 30);
    std::cout << __LINE__ << std::endl;
    string posStr = "(" + std::to_string(int(XYZ.x)) + ", " + std::to_string(int(XYZ.y)) + ", " + std::to_string(int(XYZ.z)) + ")";
    cv::putText(frame, posStr,  Point(ball_position.x + 10,ball_position.y + 20),
            FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,200,250), 1, CV_AA);
        
    std::cout << __LINE__ << std::endl;
    //end Long
    Mat frame_resized = frame.clone();
    resize(frame_resized, frame_resized, Size(frame_resized.cols/1.5, frame_resized.rows/1.5));
    imshow("Ball", frame_resized);
    cv::imwrite("/media/long/Data1/materials/thesisM1/internshipM1/TableTennis/test/Data/position1.png", frame_resized);

    char key = waitKey(0);
}


// draws the circles that found with Hough transform
void drawHoughCircles(Mat& frame, vector<Vec3f> circles) {
    for(size_t i = 0; i < circles.size(); i++)
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);

        circle(frame, center, radius, Scalar(0,0,255), 3, 8, 0);
    }
}

// draws the trajectory
void drawTrackingInfo(Mat& frame, vector<Point> positions) {
    // we draw the trajectory lines between the positions detected
    for (int j=0; j<((int)positions.size()-1); j++)
        line(frame, positions.at(j), positions.at(j+1), Scalar(0,0,255), 2, CV_AA);

    // we draw a circle for each position detected
    for (size_t j=0; j<positions.size(); j++)
        circle(frame, positions.at(j), 2, Scalar(255,0,0), 2);
}

// draws the trajectory and the rectangle of the ROI
void drawTrackingInfo(Mat& frame, vector<Point> positions, Rect roi_rect) {
    drawTrackingInfo(frame, positions);
    rectangle(frame, roi_rect, Scalar(0,255,0));
}

using namespace cv;
using namespace std;

//int main(int argc, char *argv[])
//{
//    Mat src, src_gray;

//    /// Read the image
//    src = imread("/home/parallels/Code/build-PoseEstimation-Desktop_Qt_5_13_0_GCC_64bit-Debug/Data/table.png", 1 );
//    cvtColor( src, src_gray, CV_BGR2GRAY );
//    GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );

//    vector<Vec3f> circles;
//    /// Apply the Hough Transform to find the circles
//    HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 2, 20, 81, 29, 11, 15 );
//    cout<<"Detected balls "<<circles.size()<<endl;
//    /// Draw the circles detected
//    for( size_t i = 0; i < circles.size(); i++ )
//    {
//        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
//        int radius = cvRound(circles[i][2]);
//        // circle center
//        circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
//        // circle outline
//        circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );
//    }

//    namedWindow( "Hough Circle Transform Demo", WINDOW_AUTOSIZE );
//    imshow( "Hough Circle Transform Demo", src );
//    waitKey(0);
//       cout<<"Size "<<src.size()<<endl;
//    return 0;


//    FileStorage fs;
//    fs.open("/home/parallels/Code/build-PoseEstimation-Desktop_Qt_5_13_0_GCC_64bit-Debug/Data/pose.yml", FileStorage::READ);
//    Mat A, R, t;
//    fs["A"]>>A;
//    fs["R"]>>R;
//    fs["t"]>>t;
//    cout<<"A"<<A<<endl;
//    cout<<"R"<<R<<endl;
//    cout<<"t"<<t<<endl;

//    Mat A_formalized = A.clone();
//    A_formalized.at<double>(0,2) = 0;
//    A_formalized.at<double>(1,2) = 0;
//    cout<<"A_formalized"<<endl<<A_formalized<<endl;

//    Mat deltaXYZ = (Mat_<double>(3,1) << 40, 40, 0);
//    Mat sXdeltaUV = A_formalized*deltaXYZ;
//    double s = sXdeltaUV.at<double>(0,0)/25;
//    cout<<"sXdeltaUV:"<<endl<<sXdeltaUV<<endl;
//    cout<<"s:"<<s<<endl;

//    Mat uv = s*(Mat_<double>(3,1) << 799, 730, 1);

//    cout<<"uv:"<<endl<<uv<<endl;
//    // Point in world coordinates
//    cv::Mat X_c = A.inv()*uv ; // 3x1
//    cout<<"X in cam coor:"<<endl<<X_c<<endl;

//    cv::Mat XYZ = R.inv() * ( X_c - t ); // 3x1

//    cout<<"XYZ"<<endl<<XYZ<<endl;
//    return 0;
//}
