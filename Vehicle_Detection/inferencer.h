/// Inferencer
/// By Gohur, Umair, Will
/// Cars are detected by compairing image locations with our feature 
/// representation of our car dataset. Cars that are detected will 
/// have a bounding box placed around them. 
/// Pre: Input images of cars and lane lines
/// Post: image of cars with bound boxes.
#ifndef INFERENCER_H
#define INFERENCER_H
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

class Inferencer {
public:
	std::vector<float> get_svm_detector(std::string);
	std::pair<cv::Ptr<cv::ml::SVM>, std::vector<float>> get_svm_detector(std::string, int);

	
	cv::Mat vehicle_detect(
		cv::Mat&,
		cv::HOGDescriptor&,
		std::vector<cv::Point>&,
		bool include_all_bboxes = (false)
	);
	std::vector<cv::Rect> vehicle_detect_bboxes(
		cv::Mat&,
		cv::HOGDescriptor&,
		std::vector<cv::Point>&,
		bool include_all_bboxes = (false)
	);
	std::vector<cv::Rect> vehicle_detect_bboxes(
		cv::Mat&,
		cv::HOGDescriptor&,
		std::vector<cv::Point>&,
		float bbox_confidence_threshold,
		float nms_threshold,
		bool include_all_bboxes = (false)
	);
	std::vector<cv::Rect> vehicle_detect_bboxes(
		ConfigurationParameters& config,
		cv::Mat&,
		cv::HOGDescriptor&,
		bool include_all_bboxes = (false)
	);
	std::vector<cv::Rect> respace(
		std::vector<cv::Rect>&,
		cv::Rect&
	);
	std::vector<cv::Rect> draw_bboxes(
		ConfigurationParameters& config,
		std::vector<cv::Rect>& bboxes,
		cv::Mat& img
	);
	void display_num_vehicles(cv::Mat&, std::vector<cv::Rect>&);
};
#endif
