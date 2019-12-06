#ifndef FEATURE_DESCRIPTOR_H
#define	FEATURE_DESCRIPTOR_H
#define _USE_MATH_DEFINES
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/ml.hpp>
#include <filesystem>
#include <vector>
#include <sstream>
#include <iostream>
#include <string>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "config.h"

class FeatureExtractor {
public:
	cv::Mat mask_color(cv::Mat&, std::vector<int>&,std::vector<int>&);
	cv::Mat combine_mask(cv::Mat&, cv::Mat&);

	cv::Mat remove_middle_polygons(
		cv::Mat&, 
		cv::Mat&
	);

	cv::Mat create_inner_cover_mask(
		cv::Mat&,
		std::vector<std::pair<float, float>>&,
		bool debug = (false)
	);

	cv::Mat propose_roi(
		cv::Mat&, 
		std::vector<std::pair<float, float>>&,
		bool debug = (false)
	);
	cv::Mat propose_roi(
		cv::Mat&,
		double, double,
		double, double,
		double, double,
		double, double,
		bool debug = (false)
	);

	cv::Mat lane_detect(
		ConfigurationParameters& config,
		cv::Mat&
	);
	cv::Mat lane_detect(
		cv::Mat&
	);
	cv::Mat lane_detect(
		cv::Mat&, 
		std::vector<std::pair<float, float>>&
	);
	cv::Mat lane_detect(
		cv::Mat&,
		int,
		int,
		std::vector<std::pair<float, float>>&,
		ConfigurationParameters& config,
		bool remove_between_lanes = (false)
	);

	cv::Mat extract_lane_colors(cv::Mat&);
	cv::Mat extract_lane_colors(
		ConfigurationParameters& config,
		cv::Mat&
	);

	cv::Mat get_lanes(cv::Mat&, cv::Mat&);
	cv::Mat get_lanes(cv::Mat&, cv::Mat&, int, int);
	cv::Mat get_lanes(
		ConfigurationParameters& config,
		cv::Mat&, 
		cv::Mat&, 
		int, 
		int
	);

	void draw_lane_lines(
		cv::Mat& output,
		cv::Vec4i& min_points,
		cv::Point& adjusted_right_min,
		cv::Vec4i& top_line
	);
	void draw_lane_overlay(
		cv::Mat& output,
		cv::Vec4i& min_points,
		cv::Point& adjusted_right_min,
		cv::Vec4i& top_line
	);

	cv::Vec4i find_lowest_point(
		std::vector<cv::Vec4i>&,
		int middle_pt = (650),
		bool use_middle = (true)
	);
	cv::Vec4i find_highest_point(
		std::vector<cv::Vec4i>&, 
		int middle_pt = (650),
		bool use_middle = (true)
	);
	cv::Vec4i find_lowest_point(
		std::vector<cv::Vec4i>&, 
		int l_threshold = (650),
		int r_threshold = (650)
	);
	cv::Vec4i find_highest_point(
		std::vector<cv::Vec4i>&, 
		int l_threshold = (650),
		int r_threshold = (650)
	);

	cv::Point extrapolate_line(cv::Vec4i&, int);
	void show_image(cv::Mat&,int,int,int);

	std::vector<cv::Mat> featurize_dataset(
		ConfigurationParameters& config,
		std::vector<cv::Mat>&,
		bool
	);

	cv::Mat normalize_dataset(cv::Mat&);

	std::pair<cv::Mat, cv::Mat> prepare_training_data(std::vector<cv::Mat>&,std::vector<int>&);

	std::vector<cv::Rect> sliding_window(
		cv::Mat& img, 
		cv::Size& win_stride, 
		cv::Size& window_size, 
		float scale,
		const cv::Ptr <cv::ml::SVM>& model
	);

	std::pair<std::vector<cv::Mat>, std::vector<int>> load_dataset(
		std::string, 
		std::string, 
		bool debug,
		int num_imgs = (200)
	);
	std::pair<std::vector<cv::Mat>, std::vector<int>> load_dataset(
		std::string, 
		int, 
		bool,
		int num_imgs = (200)
	);

	std::vector<std::string> split(const std::string&, char);
	std::string get_name_num(std::string&);
private:
	std::vector<cv::Mat> load_images(std::string,bool,int num_imgs = (200));
};

#endif