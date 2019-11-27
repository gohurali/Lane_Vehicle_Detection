#ifndef FEATURE_DESCRIPTOR_H
#define	FEATURE_DESCRIPTOR_H
#define _USE_MATH_DEFINES
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>
#include <filesystem>
#include <vector>
#include <sstream>
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
	std::vector<cv::Point> detection_roi(cv::Mat&,
		double, double,
		double, double,
		double, double,
		double, double,
		bool
	);

	cv::Mat get_lanes(cv::Mat&, cv::Mat&);
	cv::Mat lane_detect(cv::Mat&);
	cv::Vec4i find_lowest_point(std::vector<cv::Vec4i>&);
	cv::Vec4i find_highest_point(std::vector<cv::Vec4i>&);
	cv::Point extrapolate_line(cv::Vec4i&, int);
	void show_image(cv::Mat&,int,int,int);

	std::pair<std::vector<cv::Mat>, std::vector<int>> load_dataset(std::string,std::string,bool);
	std::vector<cv::Mat> featurize_dataset(std::vector<cv::Mat>&,bool);
	void train_svm(cv::Mat&, cv::Mat&);
	std::pair<cv::Mat, cv::Mat> prepare_training_data(std::vector<cv::Mat>&,std::vector<int>&);
	cv::Mat get_hogdescriptor_visu(const cv::Mat& color_origImg, std::vector<float>& descriptorValues, const cv::Size& size);

	std::vector<float> get_svm_detector(std::string);

private:
	std::vector<cv::Mat> load_images(std::string,bool);
	std::vector<std::string> split(const std::string&, char);
};

#endif