///////////////////////////////////////////////////////////////////////////////
//  SORT: A Simple, Online and Realtime Tracker
//  
//  This is a C++ reimplementation of the open source tracker in
//  https://github.com/abewley/sort
//  Based on the work of Alex Bewley, alex@dynamicdetection.com, 2016
//
//  Cong Ma, mcximing@sina.cn, 2016
//  
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//  
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
///////////////////////////////////////////////////////////////////////////////


#include <iostream>
#include <fstream>
#include <iomanip> // to format image names using setw() and setfill()
#include <io.h>    // to check file existence using POSIX function access(). On Linux include <unistd.h>.
#include <set>
#include <sstream>
#include "Hungarian.h"
#include "KalmanTracker.h"
#include "sort.h"
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;



//typedef struct trackingbox
//{
//	int frame;
//	int id;
//	rect_<float> box;
//}trackingbox;
//
//
//// computes iou between two bounding boxes
//double getiou(rect_<float> bb_test, rect_<float> bb_gt)
//{
//	float in = (bb_test & bb_gt).area();
//	float un = bb_test.area() + bb_gt.area() - in;
//
//	if (un < dbl_epsilon)
//		return 0;
//
//	return (double)(in / un);
//}


// global variables for counting
#define CNUM 20
int total_frames = 0;
double total_time = 0.0;

void TestSORT(string seqName, bool display);
void TestSORT1(string seqName, bool dispalyl);



int main()
{
	


	
	vector<string> sequences = { "PETS09-S2L1", "TUD-Campus", "TUD-Stadtmitte", "ETH-Bahnhof", "ETH-Sunnyday", "ETH-Pedcross2", "KITTI-13", "KITTI-17", "ADL-Rundle-6", "ADL-Rundle-8", "Venice-2" };
	for (auto seq : sequences)
		TestSORT(seq, true);
	//TestSORT("PETS09-S2L1", true);

	// Note: time counted here is of tracking procedure, while the running speed bottleneck is opening and parsing detectionFile.
	cout << "Total Tracking took: " << total_time << " for " << total_frames << " frames or " << ((double)total_frames / (double)total_time) << " FPS" << endl;

	return 0;
}



void TestSORT(string seqName, bool display)
{
	cout << "Processing " << seqName << "..." << endl;

	// 0. randomly generate colors, only for display
	RNG rng(0xFFFFFFFF);
	Scalar_<int> randColor[CNUM];
	for (int i = 0; i < CNUM; i++)
		rng.fill(randColor[i], RNG::UNIFORM, 0, 256);

	string imgPath = "D:/����/archive/2DMOT2015/train/" + seqName + "/img1/";

	if (display)
		if (_access(imgPath.c_str(), 0) == -1)
		{
			cerr << "Image path not found!" << endl;
			display = false;
		}

	// 1. read detection file
	ifstream detectionFile;
	string detFileName = "data/" + seqName + "/det.txt";
	detectionFile.open(detFileName);

	if (!detectionFile.is_open())
	{
		cerr << "Error: can not find file " << detFileName << endl;
		return;
	}

	string detLine;
	istringstream ss;
	vector<TrackingBox> detData;
	char ch;
	float tpx, tpy, tpw, tph;

	while ( getline(detectionFile, detLine) )
	{
		TrackingBox tb;

		ss.str(detLine);
		ss >> tb.frame >> ch >> tb.id >> ch;
		ss >> tpx >> ch >> tpy >> ch >> tpw >> ch >> tph;
		ss.str("");

		tb.box = Rect_<float>(Point_<float>(tpx, tpy), Point_<float>(tpx + tpw, tpy + tph));
		detData.push_back(tb);
	}
	detectionFile.close();

	// 2. group detData by frame
	int maxFrame = 0;
	for (auto tb : detData) // find max frame number
	{
		if (maxFrame < tb.frame)
			maxFrame = tb.frame;
	}

	vector<vector<TrackingBox>> detFrameData;
	vector<TrackingBox> tempVec;
	for (int fi = 0; fi < maxFrame; fi++)
	{
		for (auto tb : detData)
			if (tb.frame == fi + 1) // frame num starts from 1
				tempVec.push_back(tb);
		detFrameData.push_back(tempVec);
		tempVec.clear();
	}

	// 3. update across frames
	int frame_count = 0;
	int max_age = 1;
	int min_hits = 3;
	double iouThreshold = 0.3;
	vector<KalmanTracker> trackers;
	KalmanTracker::kf_count = 0; // tracking id relies on this, so we have to reset it in each seq.

	// variables used in the for-loop
	vector<Rect_<float>> predictedBoxes;
	vector<vector<double>> iouMatrix;
	vector<int> assignment;
	set<int> unmatchedDetections;
	set<int> unmatchedTrajectories;
	set<int> allItems;
	set<int> matchedItems;
	vector<cv::Point> matchedPairs;
	vector<TrackingBox> frameTrackingResult;
	unsigned int trkNum = 0;
	unsigned int detNum = 0;

	double cycle_time = 0.0;
	int64 start_time = 0;

	// prepare result file.
	ofstream resultsFile;
	string resFileName = "output/" + seqName + ".txt";
	resultsFile.open(resFileName);

	if (!resultsFile.is_open())
	{
		cerr << "Error: can not create file " << resFileName << endl;
		return;
	}
	//save video
	stringstream s;
	
	//s << "output/" << seqName << "." << endl;
	string videoName = "output/"+seqName+".avi";

	VideoWriter videowriter;

	//////////////////////////////////////////////
	// main loop
	for (int fi = 0; fi < maxFrame; fi++)
	{
		total_frames++;
		frame_count++;
		start_time = getTickCount();
		if (trackers.size() == 0) // the first frame met
		{
			// initialize kalman trackers using first detections.
			for (unsigned int i = 0; i < detFrameData[fi].size(); i++)
			{
				KalmanTracker trk = KalmanTracker(detFrameData[fi][i].box);
				trackers.push_back(trk);
			}
			// output the first frame detections
			for (unsigned int id = 0; id < detFrameData[fi].size(); id++)
			{
				TrackingBox tb = detFrameData[fi][id];
				resultsFile << tb.frame << "," << id + 1 << "," << tb.box.x << "," << tb.box.y << "," << tb.box.width << "," << tb.box.height << ",1,-1,-1,-1" << endl;
			}
			continue;
		}

		///////////////////////////////////////
		// 3.1. get predicted locations from existing trackers.
		predictedBoxes.clear();

		for (auto it = trackers.begin(); it != trackers.end();)
		{
			Rect_<float> pBox = (*it).predict();
			if (pBox.x >= 0 && pBox.y >= 0)
			{
				predictedBoxes.push_back(pBox);
				it++;
			}
			else
			{
				it = trackers.erase(it);
				//cerr << "Box invalid at frame: " << frame_count << endl;
			}
		}

		///////////////////////////////////////
		// 3.2. associate detections to tracked object (both represented as bounding boxes)
		// dets : detFrameData[fi]
		trkNum = predictedBoxes.size();
		detNum = detFrameData[fi].size();

		iouMatrix.clear();
		iouMatrix.resize(trkNum, vector<double>(detNum, 0));

		for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
		{
			for (unsigned int j = 0; j < detNum; j++)
			{
				// use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
				iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detFrameData[fi][j].box);
			}
		}

		// solve the assignment problem using hungarian algorithm.
		// the resulting assignment is [track(prediction) : detection], with len=preNum
		HungarianAlgorithm HungAlgo;
		assignment.clear();
		HungAlgo.Solve(iouMatrix, assignment);

		// find matches, unmatched_detections and unmatched_predictions
		unmatchedTrajectories.clear();
		unmatchedDetections.clear();
		allItems.clear();
		matchedItems.clear();

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
			trackers[trkIdx].update(detFrameData[fi][detIdx].box);
		}

		for (auto umd : unmatchedDetections)
		{
			KalmanTracker tracker = KalmanTracker(detFrameData[fi][umd].box);
			trackers.push_back(tracker);
		}

		frameTrackingResult.clear();
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

		cycle_time = (double)(getTickCount() - start_time);
		total_time += cycle_time / getTickFrequency();

		for (int i = 0; i < frameTrackingResult.size(); i++)
		{
			TrackingBox tb = frameTrackingResult[i];
			resultsFile << tb.frame << "," << tb.id << "," << tb.box.x << "," << tb.box.y << "," << tb.box.width << "," << tb.box.height << ",1,-1,-1,-1" << endl;
			//rectangle(image, tb.box, Scalar(i * 20, i * 20, i * 20), 5);
		}


		if (display) // read image, draw results and show them
		{
			
			ostringstream oss;
			oss << imgPath << setw(6) << setfill('0') << fi + 1;
			Mat img = imread(oss.str() + ".jpg");
	
			Mat hebin;
			hconcat(img, img, hebin);
			if (!videowriter.isOpened())
			{
				videowriter.open(videoName, CV_FOURCC('M', 'J', 'P', 'G'), 10, hebin.size(), 2);
				cout << hebin.size() << endl;
			}
			if (img.empty())
				continue;
			for (auto tb : frameTrackingResult)
				rectangle(hebin, tb.box, Scalar(255,255,255), 2, 8, 0);
			for (auto tb : detFrameData[fi])
			{
				rectangle(hebin, Point(tb.box.x+img.cols,tb.box.y),Point(tb.box.x + img.cols+tb.box.width, tb.box.y+tb.box.height), Scalar(0, 0, 0), 2, 8, 0);
			}
			videowriter.write(hebin);
			/*imshow(seqname, hebin);
			cvwaitkey(40);*/
		}
		
	}
	videowriter.release();
	resultsFile.close();

	/*if (display)
		destroyallwindows();*/
}
void TestSORT1(string seqName, bool display)
{
	cout << "Processing " << seqName << "..." << endl;

	// 0. randomly generate colors, only for display
	RNG rng(0xFFFFFFFF);
	Scalar_<int> randColor[CNUM];
	for (int i = 0; i < CNUM; i++)
		rng.fill(randColor[i], RNG::UNIFORM, 0, 256);

	string imgPath = "D:/����/archive/2DMOT2015/train/" + seqName + "/img1/";

	if (display)
		if (_access(imgPath.c_str(), 0) == -1)
		{
			cerr << "Image path not found!" << endl;
			display = false;
		}

	// 1. read detection file
	ifstream detectionFile;
	string detFileName = "data/" + seqName + "/det.txt";
	detectionFile.open(detFileName);

	if (!detectionFile.is_open())
	{
		cerr << "Error: can not find file " << detFileName << endl;
		return;
	}

	string detLine;
	istringstream ss;
	vector<TrackingBox> detData;
	char ch;
	float tpx, tpy, tpw, tph;

	while (getline(detectionFile, detLine))
	{
		TrackingBox tb;

		ss.str(detLine);
		ss >> tb.frame >> ch >> tb.id >> ch;
		ss >> tpx >> ch >> tpy >> ch >> tpw >> ch >> tph;
		ss.str("");

		tb.box = Rect_<float>(Point_<float>(tpx, tpy), Point_<float>(tpx + tpw, tpy + tph));
		detData.push_back(tb);
	}
	detectionFile.close();

	// 2. group detData by frame
	int maxFrame = 0;
	for (auto tb : detData) // find max frame number
	{
		if (maxFrame < tb.frame)
			maxFrame = tb.frame;
	}

	vector<vector<TrackingBox>> detFrameData;
	vector<TrackingBox> tempVec;
	for (int fi = 0; fi < maxFrame; fi++)
	{
		for (auto tb : detData)
			if (tb.frame == fi + 1) // frame num starts from 1
				tempVec.push_back(tb);
		detFrameData.push_back(tempVec);
		tempVec.clear();
	}

	// 3. update across frames
	int frame_count = 0;
	int max_age = 1;
	int min_hits = 3;
	double iouThreshold = 0.3;
	Sort sort(max_age, min_hits, iouThreshold);

	double cycle_time = 0.0;
	int64 start_time = 0;

	// prepare result file.
	ofstream resultsFile;
	string resFileName = "output/" + seqName + ".txt";
	resultsFile.open(resFileName);

	if (!resultsFile.is_open())
	{
		cerr << "Error: can not create file " << resFileName << endl;
		return;
	}
	//save video
	stringstream s;


	string videoName = "output/" + seqName + "_gai.avi";

	VideoWriter videowriter;

	for (int fi = 0; fi < maxFrame; fi++)
	{
		total_frames++;
		frame_count++;
		start_time = getTickCount();
		

		cycle_time = (double)(getTickCount() - start_time);
		total_time += cycle_time / getTickFrequency();
		vector<TrackingBox> frameTrackingResult = sort.update(detFrameData[fi]);
		for (int i = 0; i < frameTrackingResult.size(); i++)
		{
			TrackingBox tb = frameTrackingResult[i];
			resultsFile << tb.frame << "," << tb.id << "," << tb.box.x << "," << tb.box.y << "," << tb.box.width << "," << tb.box.height << ",1,-1,-1,-1" << endl;
			//rectangle(image, tb.box, Scalar(i * 20, i * 20, i * 20), 5);
		}


		if (display) // read image, draw results and show them
		{

			ostringstream oss;
			oss << imgPath << setw(6) << setfill('0') << fi + 1;
			Mat img = imread(oss.str() + ".jpg");

			Mat hebin;
			hconcat(img, img, hebin);
			if (!videowriter.isOpened())
			{
				videowriter.open(videoName, CV_FOURCC('M', 'J', 'P', 'G'), 10, hebin.size(), 2);
				cout << hebin.size() << endl;
			}
			if (img.empty())
				continue;
			for (auto tb : frameTrackingResult)
				rectangle(hebin, tb.box, Scalar(255, 255, 255), 2, 8, 0);
			for (auto tb : detFrameData[fi])
			{
				rectangle(hebin, Point(tb.box.x + img.cols, tb.box.y), Point(tb.box.x + img.cols + tb.box.width, tb.box.y + tb.box.height), Scalar(0, 0, 0), 2, 8, 0);
			}
			videowriter.write(hebin);
			/*imshow(seqname, hebin);
			cvwaitkey(40);*/
		}

	}
	videowriter.release();
	resultsFile.close();

	/*if (display)
		destroyallwindows();*/
}
