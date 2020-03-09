#ifndef _KCFTRACKERPOOL_HEADERS
#define _KCFTRACKERPOOL_HEADERS
#include "kcftracker.hpp"
#include "opencv2/tracking.hpp"
#include "PosEstimation.h"
#include "opencv2/objdetect.hpp"
#include <memory>
#include <omp.h>
#include <thread>
#include <ctime>
#include <sys/time.h>
#include <iostream>

#define MAX_TIME_TO_LIVE 30
#define SCORE_DETECT_SUCCESS 8
#define MINUS_CORNER 3
#define MIN_MINUS_SCORE_TRACKING 1
#define MAX_MINUS_SCORE_TRACKING 3
#define THRESHOLD_TRACKING 0.2

namespace kcf
{
class KCFTrackPool
{
public:
    cv::Rect roiDetect;
    std::vector<KCFTracker *> track_pool;
    unsigned long long last_track_id = 0;
    std::vector<TrackCache> caches;
    cv::CascadeClassifier cascade;
    PosEstimation* estimation;
    std::vector<cv::String> fn;
    int countFrame = 0;
public:
    KCFTrackPool(){}

    ~KCFTrackPool(){}

    unsigned long long genID() {

    }

    std::vector<KCFTracker*> getListTrack() {return track_pool;}

    int initCascade(std::string cascade_name){
        if (!cascade.load(cascade_name))
        {
            std::cout << "--(!)Error loading face cascade\n";
            return -1;
        };
        return 1;
    }

    void initEstimation(std::string posePath){
        estimation = new PosEstimation(posePath);
    }

    void reTrackId(KCFTracker& track) {

    }

    int findMatchedTrack_old(cv::Rect2d rect) {
        int res = -1;
        if (track_pool.size() > 0) {
            for (size_t i = 0; i < track_pool.size(); ++i) {
                if (track_pool[i]->isMatchedWithNewTrack(rect)) {
                    res = i;
                    break;
                }
            }
        }
        return res;
    }

    int findMatchedTrack(cv::Rect rect) {
        int res = -1;
        double maxOverlap = 0.15;
        if (track_pool.size() > 0) {
            // 1. kiem tra overlap
            for (size_t i = 0; i < track_pool.size(); ++i) {
                double overlap = track_pool[i]->overlapWithNewTrack(rect);
                if (overlap>maxOverlap) {
                    maxOverlap = overlap;
                    res = i;
                }
            }
        }
        return res;
    }

    // remove track when timeToLive = 0
    void removeByDetection()
    {
        auto lst = this->getListTrack();
        for (int j=0; j<lst.size(); j++) {
            if(lst[j]->timeToLive < 1) {
                lst[j]->need_delete = true;
                deleteTrackById(lst[j]->_track_id);
            }
        }
    }

    void updateTracksVelocity(){
        for (int i = 0; i < track_pool.size(); i++) {
            cv::Point2f centerSource = cv::Point2f(track_pool[i]->_prev_roi.x + track_pool[i]->_prev_roi.width/2, track_pool[i]->_prev_roi.y + track_pool[i]->_prev_roi.height/2);
            cv::Point2f centerDestination = cv::Point2f(track_pool[i]->_current_roi.x + track_pool[i]->_current_roi.width/2, track_pool[i]->_current_roi.y + track_pool[i]->_current_roi.height/2);
            track_pool[i]->velocity = track_pool[i]->velocity * KEEP_VELOCITY_COEFF + (centerDestination - centerSource) * (1 - KEEP_VELOCITY_COEFF);
            track_pool[i]->_prev_roi = track_pool[i]->_current_roi;
        }
    }

    void updatePredictRoi(){
        for (int i = 0; i < track_pool.size(); i++) {

            track_pool[i]->_current_roi.x += track_pool[i]->velocity.x*PREDICT_VELOCITY_COEFF;
            track_pool[i]->_current_roi.y += track_pool[i]->velocity.y*PREDICT_VELOCITY_COEFF;
        }
    }

    // find matched track of bbox list
    std::vector<int> findMatchedTrack(std::vector<cv::Rect> rectList)
    {
        std::vector<int> match_List;
        int match_id;
        for (int i = 0; i < rectList.size(); i++)
        {
            match_id = -1;
            double maxOverlap = 0.15;
            if (track_pool.size() > 0) {
                // 1. kiem tra overlap
                for (int k = 0; k < track_pool.size(); k++) {
                    double overlap = track_pool[k]->overlapWithNewTrack(rectList[i]);
                    if (overlap>maxOverlap) {
                        double maxOverlapTemp = 0.15;
                        for(int j = 0; j < rectList.size(); j++)
                        {
                            double overlapTemp = track_pool[k]->overlapWithNewTrack(rectList[j]);
                            if(overlapTemp > maxOverlapTemp)
                            {
                                maxOverlapTemp = overlapTemp;
                            }
                        }
                        if (overlap >= maxOverlapTemp)
                        {
                            maxOverlap = overlap;
                            match_id = k;
                        }
                    }
                }
            }

            match_List.push_back(match_id);
        }
        return match_List;
    }

    void updateTrackCorner(cv::Size imageSize){
        if(!(imageSize.width == 0 && imageSize.height == 0)){
            std::vector<KCFTracker*> lst = this->getListTrack();
            for (int i=0; i<lst.size(); i++)
            {
                if(lst[i]->isCorner == false){
                    cv::Rect2d rect = lst[i]->_current_roi;
                    if(rect.x <= 10 || rect.y <= 10 ||
                            (abs(imageSize.width - (rect.x + rect.width))) <= 10 ||
                            (abs(imageSize.height - (rect.y + rect.height))) <= 10)
                    {
                        lst[i]->isCorner = true;
                        lst[i]->timeToLive -= MINUS_CORNER;
                    }
                }else{
                    cv::Rect2d rect = lst[i]->_current_roi;
                    if(rect.x > 10 || rect.y > 10 ||
                            (abs(imageSize.width - (rect.x + rect.width))) > 10 ||
                            (abs(imageSize.height - (rect.y + rect.height))) > 10)
                    {
                        lst[i]->isCorner = false;
                    }else{
                        lst[i]->timeToLive -= MINUS_CORNER;
                    }
                }
            }
        }
    }

    void putTrackToPool(const cv::Mat frame, cv::Rect rect, int match_id) {
        bool HOG = false, FIXEDWINDOW = true, MULTISCALE = true, LAB = false, DSST = false; //LAB color space features

        if (match_id > -1) {
            //Old track
            track_pool[match_id]->_current_roi = rect;
            track_pool[match_id]->isUpdateRoiByDetect = true;
            track_pool[match_id]->timeToLive += SCORE_DETECT_SUCCESS;
            if (track_pool[match_id]->countOverlap < 2) {
                track_pool[match_id]->countOverlap++;
            }
        } else {
            //New track
            last_track_id++;
            auto new_kcf_track = new KCFTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);
            //------------------------------------------------------
            //Generate unique Id
            struct timeval tp;
            gettimeofday(&tp, NULL);
            unsigned long long ms = (unsigned long long)tp.tv_sec * 1000 + tp.tv_usec / 1000;
            //------------------------------------------------------
            new_kcf_track->init(frame,rect, ms*10+last_track_id%10);
            new_kcf_track->timeToLive = SCORE_DETECT_SUCCESS;

            track_pool.push_back(new_kcf_track);
        }
    }

    //Input: detect bbox,
    void putTrackToPool(const cv::Mat frame, cv::Rect2d rect) {
        int match_id = findMatchedTrack(rect);
        bool HOG = false, FIXEDWINDOW = true, MULTISCALE = false, LAB = false, DSST = false; //LAB color space features

        if (match_id > -1) {
            //Old track
            track_pool[match_id]->_current_roi = rect;
            track_pool[match_id]->isUpdateRoiByDetect = true;
        }
        else {
            //New track
            last_track_id++;
            auto new_kcf_track = new KCFTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);
            //------------------------------------------------------
            //Generate unique Id
            struct timeval tp;
            gettimeofday(&tp, NULL);
            
            unsigned long long ms = (unsigned long long)tp.tv_sec * 1000 + tp.tv_usec / 1000;
            //------------------------------------------------------
            new_kcf_track->init(frame,rect, ms*10+last_track_id%10);
            track_pool.push_back(new_kcf_track);
        }
    }

    void removeTrackFromPool() {
        if(track_pool.size() > 0) {
            track_pool.erase(track_pool.end() - 1);
        }
    }
    
    //0 -> 1.0 -50 -1
    // test parallel update
    void updateTrack(const cv::Mat frame) {
        std::vector<std::thread> threadPool;
        std::for_each(track_pool.begin(), track_pool.end(),
                      [&](KCFTracker* ptr){
            threadPool.emplace_back([ptr, &frame]() {
                float confidenceKCF = ptr->update(frame);
                int score = 0;
                if (confidenceKCF > 1) {
                    score = -MIN_MINUS_SCORE_TRACKING;
                } else if(confidenceKCF < THRESHOLD_TRACKING){
                    score = -MAX_MINUS_SCORE_TRACKING;
                }else{
                    score = -((MIN_MINUS_SCORE_TRACKING - MAX_MINUS_SCORE_TRACKING)*confidenceKCF + MAX_MINUS_SCORE_TRACKING);
//                    std::cout << "confidence kcf : " << confidenceKCF << ", score : " << score << std::endl;
                }

                ptr->timeToLive += score;
                if(ptr->timeToLive < 1){
                    ptr->need_delete = true;
                }
            });
        }
        );
        for (auto &t : threadPool)
            t.join();

        // xoa phuc
        for (auto track : track_pool) {
            if(track->timeToLive > MAX_TIME_TO_LIVE){
                track->timeToLive = MAX_TIME_TO_LIVE;
            }
            if (track->need_delete) {
                deleteTrackById(track->_track_id);
            }
        }
    }

    void updateTrackBeforeDetect(const cv::Mat frame) {
        std::vector<std::thread> threadPool;
        std::for_each(track_pool.begin(), track_pool.end(),
                      [&](KCFTracker* ptr){
            threadPool.emplace_back([ptr, &frame]() {
                float confidenceKCF = ptr->update(frame);
            });
        }
        );
        for (auto &t : threadPool)
            t.join();
    }

    int findTrackById(unsigned long long track_id){
        if (track_pool.size() == 0)
            return -1;
        auto target =
                std::find_if(track_pool.begin(), track_pool.end(),
                             [&, track_id](KCFTracker* ptr) -> bool {
            return ptr->_track_id == track_id;
        });

        if (target == track_pool.end())
            return -1;
        else
            return target - track_pool.begin();
    }

    void drawRectTrack(cv::Mat &frm){
        for(int i = 0; i < track_pool.size(); ++i){
            auto displayRect = track_pool[i]->_current_roi;
            cv::rectangle(frm, displayRect, cv::Scalar(0, 255, 0), 2);
            // cv::putText(frm, std::to_string(track_pool[i]->_track_id), displayRect.tl(), 0, 0.8, cv::Scalar(0, 0, 255), 2);
            // cv::putText(frm, "frmToLive: " + std::to_string(track_pool[i]->timeToLive), cv::Point(displayRect.x, displayRect.y + displayRect.height * 0.5), 0, 0.8, cv::Scalar(0, 255, 0), 2);
        }
    }

    void drawPosition(cv::Mat &frm){
        for(int i = 0; i < track_pool.size(); ++i){
            auto displayRect = track_pool[i]->_current_roi;
            cv::Point3d realPt = estimation->convertFrom2D(cv::Point2d(displayRect.x + displayRect.width/2, displayRect.y + displayRect.height/2),
             (displayRect.width + displayRect.height)/4);
            std::string displayText = "(" + std::to_string(realPt.x/1000) + "," + std::to_string(realPt.y/1000) + "," + std::to_string(realPt.z/1000) + ")";
            cv::putText(frm, displayText, displayRect.tl(), 0, 0.8, cv::Scalar(0, 0, 255), 2);
        }
    }

    void drawPosition(cv::Mat &frm, std::vector<cv::Point3d> &outPosition){
        for(int i = 0; i < track_pool.size(); ++i){
            auto displayRect = track_pool[i]->_current_roi;
            cv::Point3d realPt;
            cv::Point2d centerPoint(displayRect.x + displayRect.width/2, displayRect.y + displayRect.height/2);
            #if 0
            realPt = estimation->convertFrom2D(centerPoint, (displayRect.width + displayRect.height)/8);
            #else
            realPt = estimation->convertFrom2DS(centerPoint, 1000);
            #endif
            if(centerPoint.x > roiDetect.x && centerPoint.x < roiDetect.x + roiDetect.width 
            && centerPoint.y > roiDetect.y && centerPoint.y < roiDetect.y + roiDetect.height){
                outPosition.push_back(realPt);
            }

            std::string displayText = "(" + std::to_string(int(realPt.x/10)) + "," + std::to_string(int(realPt.y/10)) + "," + std::to_string(int(realPt.z/10)) + ")";
            cv::Point displayPoint = displayRect.tl();
            displayPoint.y -= 20;
            // cv::putText(frm, displayText, displayPoint, 0, 1.5, cv::Scalar(0, 0, 255), 4);
        }
    }

    void deleteTrackById(unsigned long long track_id){
        std::vector<KCFTracker*> listDeleteTrack;
        for (int i = 0; i < track_pool.size(); ++i) {
            if (track_pool[i]->_track_id == track_id) {
                listDeleteTrack.push_back(track_pool[i]);
            }
        }
        track_pool.erase(
                    std::remove_if
                    (
                        track_pool.begin(),
                        track_pool.end(),
                        [&, track_id](KCFTracker* ptr){
            auto isDel = ptr->_track_id == track_id;
            return isDel;
        }
        ),track_pool.end());
        for (int i = 0; i < listDeleteTrack.size(); ++i) {
            if (listDeleteTrack[i])
            {
                delete listDeleteTrack[i];
            }
            
        }
    }
};
}
#endif
