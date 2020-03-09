#ifndef DEFINE_H
#define DEFINE_H

#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;


#define SHOW_WINDOWS  // if defined, display a window with image result for each step

// we use only orange balls, like RGB = [255, 252, 31]
// For this color, HSV is [30, 224, 255]
// We want to ignore the V of HSV, because we don't want to be dependent
// of the amount of light
//const int h = 128, h_threshold = 128;
const Scalar BALL_COLOR_HSV_MIN(0, 0, 210);
const Scalar BALL_COLOR_HSV_MAX(180, 25, 255);

// the size of the kernels used for blurring / morphological operations
const int BLUR_KERNEL_LENGTH = 15;
const int CLOSING_KERNEL_LENGTH = 51;

// the size of the zone we will use to search for the ball,
// if we know the position of the ball in the previous frame
const int BALL_SIZE = 10;  // pixels
const int ROI_WIDTH  = BALL_SIZE * 15;
const int ROI_HEIGHT = BALL_SIZE * 7;

const Point GT_CENTER(801, 726);
const int GT_RADIUS = 12;

#endif
