#ifndef FEATURE_DESCRIPTOR_H
#define	FEATURE_DESCRIPTOR_H
#define _USE_MATH_DEFINES
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <math.h>
class FeatureExtractor {
public:
	cv::Mat mask_color(cv::Mat&, std::vector<int>&,std::vector<int>&);
	cv::Mat combine_mask(cv::Mat&, cv::Mat&);
	cv::Mat propose_roi(cv::Mat&,
		double, double,
		double, double,
		double, double,
		double, double
	);
	cv::Mat get_lanes(cv::Mat&, cv::Mat&);
	cv::Mat lane_detect(cv::Mat&);
	void show_image(cv::Mat&,int,int,int);
};

#endif