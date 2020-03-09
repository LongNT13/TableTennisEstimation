#include <iostream>
#include <chrono>
#include <unistd.h>
#include <thread>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "trackpool.hpp"

using namespace std;
using namespace cv;
#define DETECT_INTERVAL 2
kcf::KCFTrackPool track_manager[2];

std::string cascade_name = "../model/1/17stages/cascade.xml";
std::string imagesPath = "../Cam1/newimage/*.png";
std::string posePath = "Data/pose1.yml";
std::string windowName = "cam1";
std::string cascade_name2 = "../model/1/20stages/cascade.xml";
std::string imagesPath2 = "../Cam2/newimage/*.png";
std::string posePath2 = "Data/pose2.yml";
std::string windowName2 = "cam2";

void detectAndTrack(Mat &frm, int index, std::vector<cv::Point3d> &outputPositions);

void initRoi(int x, int y, int width, int height, int index)
{
    track_manager[index].roiDetect.x = x;
    track_manager[index].roiDetect.y = y;
    track_manager[index].roiDetect.width = width;
    track_manager[index].roiDetect.height = height;
}

int init(std::string posePath, std::string imagesPath, std::string cascade_name, int index)
{
    int check = 1;
    track_manager[index].initEstimation(posePath);
    check = track_manager[index].initCascade(cascade_name);

    std::cout << "start reading images" << std::endl;
    glob(imagesPath, track_manager[index].fn, false);
    std::cout << "finish reading images" << std::endl;
    return check;
}

cv::Mat getFrame(int index, int i)
{
    cv::Mat frame;
    std::cout << track_manager[index].fn[i] << std::endl;
    frame = imread(track_manager[index].fn[i]);

    if (frame.empty())
    {
        cout << "--(!) No captured frame -- Break!\n";
    }
    return frame;
}

// int main(int argc, char const *argv[])
// {

//     Mat rMatrix, tMatrix, aMatrix, pMatrix;
//     Mat rMatrix2, tMatrix2, aMatrix2, pMatrix2;
//     readFs("Data/pose1.yml", rMatrix, tMatrix, aMatrix, pMatrix);
//     readFs("Data/pose2.yml", rMatrix2, tMatrix2, aMatrix2, pMatrix2);

//     cv::Point2d uv(206, 640);
//     cv::Point2d uv2(1852, 682);
//     int option = 3;
//     if (option == 0)
//     {
//         cv::Point3d camPos1 = calCamPosition(rMatrix, tMatrix);
//         cv::Point3d camPos2 = calCamPosition(rMatrix2, tMatrix2);
//         std::cout << "cam pos1: " << camPos1 << std::endl;
//         std::cout << "cam pos2: " << camPos2 << std::endl;
//     }
//     else if (option == 1)
//     {
//         cv::Point2d tempPoint = from3dTo2d(cv::Point3d(0,0,0), pMatrix, aMatrix);
//         cv::Point2d tempPoint2 = from3dTo2d(cv::Point3d(0,0,0), pMatrix2, aMatrix2);
//         std::cout << "cam1 0 0 0 to 2d: " << tempPoint << std::endl;
//         std::cout << "cam2 0 0 0 to 2d: " << tempPoint2 << std::endl;
//     }
//     else if (option == 2)
//     {
//         //calculate object position 3d second point
//         cv::Point3d xyz = convertFrom2D(uv, aMatrix, rMatrix, tMatrix);
//         cv::Point3d xyz2 = convertFrom2D(uv2, aMatrix2, rMatrix2, tMatrix2);
//         std::cout << "xyz : " << xyz << std::endl;
//     }
//     else if (option == 3)
//     {
//         cv::Point3d camPos1 = calCamPosition(rMatrix, tMatrix);
//         cv::Point3d camPos2 = calCamPosition(rMatrix2, tMatrix2);
//         cv::Point3d xyz = convertFrom2D(uv, aMatrix, rMatrix, tMatrix);
//         cv::Point3d xyz2 = convertFrom2D(uv2, aMatrix2, rMatrix2, tMatrix2);
//         float distance;
//         cv::Point3f temp;
//         analyze2Line(camPos1, xyz, camPos2, xyz2, distance, temp);
//         std::cout << "distance : " << distance << std::endl;
//         std::cout << "point : " << temp << std::endl;
//     }
//     return 0;
// }

int main(int argc, const char **argv)
{
    VideoWriter video("out.avi",CV_FOURCC('M','J','P','G'), 10, Size(2048, 1088),true);
    VideoWriter video2("out2.avi",CV_FOURCC('M','J','P','G'), 10, Size(2048, 1088),true);

    cv::namedWindow(windowName, WINDOW_GUI_EXPANDED);
    cv::namedWindow(windowName2, WINDOW_GUI_EXPANDED);
    init(posePath, imagesPath, cascade_name, 0);
    init(posePath2, imagesPath2, cascade_name2, 1);
    initRoi(0, 300, 2048, 788, 0);
    initRoi(600, 0, 1448, 1088, 1);
    for (size_t i = 0; i < 1000; i++)
    {
        cv::Mat frmCam1 = getFrame(0, i);
        cv::Mat frmCam2 = getFrame(1, i);
        //-- 3. Apply the classifier to the frame
        std::vector<cv::Point3d> posForCam1, posForCam2;
        detectAndTrack(frmCam1, 0, posForCam1);
        detectAndTrack(frmCam2, 1, posForCam2);

        float minDistance = 0;
        cv::Point3f position2Cam(0, 0, 0);
        bool isFirst = true;
        cv::Point3d camPos1 = track_manager[0].estimation->getCamPosition();
        cv::Point3d camPos2 = track_manager[1].estimation->getCamPosition();
        for (size_t i = 0; i < posForCam1.size(); i++)
        {
            for (size_t j = 0; j < posForCam2.size(); j++)
            {
                float distance;
                cv::Point3f tempPoint;
                cv::Point3d xyz = posForCam1[i];
                cv::Point3d xyz2 = posForCam2[j];
                analyze2Line(camPos1, xyz, camPos2, xyz2, distance, tempPoint);
                if (isFirst)
                {
                    minDistance = distance;
                    position2Cam = tempPoint;
                    isFirst = false;
                }
                else
                {
                    if(distance < minDistance){
                        minDistance = distance;
                        position2Cam = tempPoint;
                    }
                }
            }
        }
        std::cout << "mindistance : " << minDistance << std::endl;
        std::cout << "Point : " << position2Cam << std::endl;
        std::string displayText = "(" + std::to_string(int(position2Cam.x/10)) + ", " + std::to_string(int(position2Cam.y/10)) + ", " + std::to_string(int(position2Cam.z/10)) + ")";

        cv::Point displayPoint = track_manager[0].track_pool[0]->_current_roi.tl();
        displayPoint.y -= 20;
        cv::Point displayPoint2 = track_manager[1].track_pool[0]->_current_roi.tl();
        displayPoint2.y -= 20;
        cv::putText(frmCam1, displayText, displayPoint, 0, 1.5, cv::Scalar(0, 0, 255), 4);
        cv::putText(frmCam2, displayText, displayPoint2, 0, 1.5, cv::Scalar(0, 0, 255), 4);
        imshow(windowName, frmCam1);
        imshow(windowName2, frmCam2);

        video.write(frmCam1);
        video2.write(frmCam2);
        if (waitKey(60) == 27)
        {
            break; // escape
        }
    }

    // while (1)
    // {
    //     std::this_thread::sleep_for(std::chrono::seconds(1));
    // }
    return 0;
}
void detectAndTrack(Mat &frm, int index, std::vector<cv::Point3d> &outputPositions)
{
    Mat frame_gray;
    cvtColor(frm, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    // update PredictRoi
    track_manager[index].updatePredictRoi();

    //-- Detect balls
    std::vector<cv::Rect> balls;
    if ((track_manager[index].countFrame++) % DETECT_INTERVAL == 0)
    {
        //step 0. update track
        track_manager[index].updateTrackBeforeDetect(frm);
        //====================================================================

        auto start = std::chrono::steady_clock::now();
        track_manager[index].cascade.detectMultiScale(frame_gray, balls, 1.1, 6, 0);
        std::cout << "Time needed : " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() << std::endl;
        
        std::vector<int> deleteBall;
        for (size_t i = 0; i < balls.size(); i++)
        {
            if(!track_manager[index].roiDetect.contains(cv::Point(balls[i].x + balls[i].width/2, balls[i].y + balls[i].height/2))){
                deleteBall.push_back(i);
            }
        }
        auto it = balls.begin();
        int count = 0;
        while (it != balls.end())
        {
            // remove odd numbers
            bool isDel = false;
            for (size_t i = 0; i < deleteBall.size(); i++)
            {
                if(count == deleteBall[i]){
                    isDel = true;
                    break;
                }
            }
            
            if (isDel) {
                // erase() invalidates the iterator, use returned iterator
                it = balls.erase(it);
            } else {
                ++it;
            }
            count ++;
        }
        
        for (size_t i = 0; i < balls.size(); i++)
        {
            cv::rectangle(frm, cv::Rect(balls[i].x - 5, balls[i].y - 5, balls[i].width + 10, balls[i].height + 10), cv::Scalar(0, 0, 255), 4);
        }

        std::vector<int> matchedList = track_manager[index].findMatchedTrack(balls);
        for (int i = 0; i < matchedList.size(); i++)
        {
            track_manager[index].putTrackToPool(frm, balls[i], matchedList[i]);
        }

        //remove track that not belong to any detect bbox (for n frames)
        track_manager[index].updateTrackCorner(cv::Size(frm.cols, frm.rows));
    }

    //====================================================================
    // Step 3. update track
    track_manager[index].updateTrack(frm);

    // update velocity
    track_manager[index].updateTracksVelocity();

    //draw
    track_manager[index].drawRectTrack(frm);
    track_manager[index].drawPosition(frm, outputPositions);
    //====================================================================
    // std::vector<kcf::KCFTracker *> listTrackKcf = track_manager[index].getListTrack();
}
