#include "feature_extractor.h"

cv::Mat FeatureExtractor::mask_color(cv::Mat& img, std::vector<int>& lower_b, std::vector<int>& upper_b) {
	cv::Mat lower_bound(lower_b);
	cv::Mat upper_bound(upper_b);
	cv::Mat mask;
	cv::inRange(img, lower_bound, upper_bound,mask);
	return mask;
}

cv::Mat FeatureExtractor::combine_mask(cv::Mat& mask1, cv::Mat& mask2) {
	cv::Mat combined;
	cv::bitwise_or(mask1, mask2, combined);
	return combined;
}