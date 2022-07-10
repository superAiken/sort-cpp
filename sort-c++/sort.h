#ifndef SORT_H
#define SORT_H
#include <vector>
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
double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
	float in = (bb_test & bb_gt).area();
	float un = bb_test.area() + bb_gt.area() - in;

	if (un < DBL_EPSILON)
		return 0;

	return (double)(in / un);
}
#include "KalmanTracker.h"
class Sort
{
public:
	Sort(int max_age,int min_hits, double iouThreshold);
	~Sort();
	TrackingBox update();

private:
	int max_age;
	int min_hits;
	vector<KalmanTracker> trackers;
	int frame_count;
};

#endif
