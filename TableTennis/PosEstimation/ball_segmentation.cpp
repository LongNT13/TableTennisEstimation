#include "ball_segmentation.h"

using namespace std;
using namespace cv;


/**
    Apply a segmentation with a threshold based on the color of the ball

    @param roi_bgr_blurred The ROI blurred, formatted as BGR
    @param roi_binarized The output binarized frame
*/
void thresholdSegmentation(Mat& roi_bgr_blurred, Mat& roi_binarized) {
    Mat roi_hsv(Size(roi_bgr_blurred.cols, roi_bgr_blurred.rows), CV_8UC3);

    // we convert the frame to HSV
    cvtColor(roi_bgr_blurred, roi_hsv, CV_BGR2HSV);

    // we compute the mask by binarizing the picture with a color threshold
    inRange(roi_hsv, BALL_COLOR_HSV_MIN, BALL_COLOR_HSV_MAX, roi_binarized);

    //cout<<"Size of bin "<<roi_binarized.size()<<endl;

    #ifdef SHOW_WINDOWS
        Mat roi_resized = roi_binarized.clone();
        cv::circle(roi_resized, GT_CENTER, GT_RADIUS, cv::Scalar(255,255,0),1);
        resize(roi_resized, roi_resized, Size(roi_resized.cols/1.5, roi_resized.rows/1.5));
        imshow("threshold segmentation", roi_resized);
    #endif

    // we apply a closing (dilatation then erosion)
    morphologyEx(roi_binarized, roi_binarized, MORPH_CLOSE,
                getStructuringElement(MORPH_ELLIPSE, Size(CLOSING_KERNEL_LENGTH,CLOSING_KERNEL_LENGTH)));
}


