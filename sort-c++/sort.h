#ifndef SORT_H
#define SORT_H
#include <vector>
#include <set>
#include "KalmanTracker.h"
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace std;
using namespace cv;

typedef struct TrackingBox
{
	int frame;
	int id;
	Rect_<float> box;
}TrackingBox;


// Computes IOU between two bounding boxes
double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt);

class Sort
{
public:
	Sort(int,int,double);
	vector<TrackingBox> update(vector<TrackingBox>);

private:
	int max_age;
	int min_hits;
	vector<KalmanTracker> trackers;
	int frame_count;
	double iouThreshold;
	vector<Rect_<float>> predictedBoxes;
	vector<vector<double>> iouMatrix;
	vector<int> assignment;
	set<int> unmatchedDetections;
	set<int> unmatchedTrajectories;
	set<int> allItems;
	set<int> matchedItems;
	vector<cv::Point> matchedPairs;
	vector<TrackingBox> frameTrackingResult;
	unsigned int trkNum ;
	unsigned int detNum ;
};

#endif
