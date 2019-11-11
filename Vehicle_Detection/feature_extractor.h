#ifndef FEATURE_DESCRIPTOR_H
#define	FEATURE_DESCRIPTOR_H
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <stdio.h>
class FeatureExtractor {
public:
	cv::Mat mask_color(cv::Mat&, std::vector<int>&,std::vector<int>&);
	cv::Mat combine_mask(cv::Mat&, cv::Mat&);
};

#endif