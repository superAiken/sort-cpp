#include "sort.h"
#include "Hungarian.h"
double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
	float in = (bb_test & bb_gt).area();
	float un = bb_test.area() + bb_gt.area() - in;

	if (un < DBL_EPSILON)
		return 0;
	return (double)(in / un);
}
Sort::Sort(int a, int b, double c)
{
	max_age = a;
	min_hits = b;
	iouThreshold = c;
	detNum = 0;
	trkNum = 0;
	frame_count = 0;
}
vector<TrackingBox> Sort::update(vector<TrackingBox> detBoxes)
{
	predictedBoxes.clear();
	iouMatrix.clear();
	assignment.clear();
	unmatchedDetections.clear();
	unmatchedTrajectories.clear();
	allItems.clear();
	matchedItems.clear();
	matchedPairs.clear();
	frameTrackingResult.clear();

	if (trackers.size() == 0)
	{
		for (int i = 0; i < detBoxes.size(); i++)
		{
			KalmanTracker tb = KalmanTracker(detBoxes[i].box);
			trackers.push_back(tb);
		}
		return detBoxes;
	}

	for (auto it = trackers.begin(); it != trackers.end();)
	{
		Rect_<float> pBox = (*it).predict();
		if (pBox.x >= 0 && pBox.y >= 0)
		{
			predictedBoxes.push_back(pBox);
			it++;
		}
		else {
			it = trackers.erase(it);
		}
	}
	trkNum = predictedBoxes.size();
	detNum = detBoxes.size();
	iouMatrix.resize(trkNum, vector<double>(detNum, 0));
	for (unsigned int i = 0; i < trkNum; i++)
	{
		for (unsigned int j = 0; j < detNum; j++)
		{
			iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detBoxes[j].box);
		}
	}
	HungarianAlgorithm HungAlgo;
	HungAlgo.Solve(iouMatrix, assignment);
	if (detNum > trkNum) //	there are unmatched detections
	{
		for (unsigned int n = 0; n < detNum; n++)
			allItems.insert(n);

		for (unsigned int i = 0; i < trkNum; ++i)
			matchedItems.insert(assignment[i]);

		set_difference(allItems.begin(), allItems.end(),
			matchedItems.begin(), matchedItems.end(),
			insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
	}
	else if (detNum < trkNum) // there are unmatched trajectory/predictions
	{
		for (unsigned int i = 0; i < trkNum; ++i)
			if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
				unmatchedTrajectories.insert(i);
	}
	else
		;

	matchedPairs.clear();
	for (unsigned int i = 0; i < trkNum; ++i)
	{
		if (assignment[i] == -1) // pass over invalid values
			continue;
		if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
		{
			unmatchedTrajectories.insert(i);
			unmatchedDetections.insert(assignment[i]);
		}
		else
			matchedPairs.push_back(cv::Point(i, assignment[i]));
	}


	int detIdx, trkIdx;
	for (unsigned int i = 0; i < matchedPairs.size(); i++)
	{
		trkIdx = matchedPairs[i].x;
		detIdx = matchedPairs[i].y;
		trackers[trkIdx].update(detBoxes[detIdx].box);
	}

	for (auto umd : unmatchedDetections)
	{
		KalmanTracker tracker = KalmanTracker(detBoxes[umd].box);
		trackers.push_back(tracker);
	}
	for (auto it = trackers.begin(); it != trackers.end();)
	{
		if (((*it).m_time_since_update < 1) &&
			((*it).m_hit_streak >= min_hits || frame_count <= min_hits))
		{
			TrackingBox res;
			res.box = (*it).get_state();
			res.id = (*it).m_id + 1;
			res.frame = frame_count;
			frameTrackingResult.push_back(res);
			it++;
		}
		else
			it++;

		if (it != trackers.end() && (*it).m_time_since_update > max_age)
			it = trackers.erase(it);
	}
	return frameTrackingResult;
}