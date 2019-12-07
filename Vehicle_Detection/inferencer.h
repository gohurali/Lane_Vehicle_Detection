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
	
	/// get_svm_detector
	/// Opens the support vector machine serialized file
	/// Preconditions:		The string location of the serialized model file
	/// Postconditions:		A pair including a pointer to the SVM model and 
	///				vector of float coefficients of the SVM model
	std::vector<float> get_svm_detector(std::string);
	
	/// get_svm_detector
	/// Opens the support vector machine serialized file
	/// Preconditions:		The string location of the serialized model file
	/// Postconditions:		A pair including a pointer to the SVM model and 
	///				vector of float coefficients of the SVM model
	std::pair<cv::Ptr<cv::ml::SVM>, std::vector<float>> get_svm_detector(std::string, int);

	/// Search for comparitable features, place bounding box, use Non max suppression to 
	/// limit the number of bounding boxes per vehicle.
	/// Preconditions:	The image to draw on, HOG loaded with SVM also needs to be
	///			loaded into memory, and ROI for sliding window needs to be
	///			defined.
	/// Postconditions: img with bboxes is returned
	cv::Mat vehicle_detect(
		cv::Mat&,
		cv::HOGDescriptor&,
		std::vector<cv::Point>&,
		bool include_all_bboxes = (false)
	);
	
	/// vehicle_detect_bboxes
	/// Rather than drawing the bboxes, the vector of bboxes
	/// is returned.
	/// Preconditions:	The image to draw on, HOG loaded with SVM also needs to be
	///			loaded into memory, and ROI for sliding window needs to be
	///			defined.
	/// Postconditions: 	Vector of bboxes (cv::Rect) is returned
	std::vector<cv::Rect> vehicle_detect_bboxes(
		cv::Mat&,
		cv::HOGDescriptor&,
		std::vector<cv::Point>&,
		bool include_all_bboxes = (false)
	);
	
	/// vehicle_detect_bboxes
	/// Rather than drawing the bboxes, the vector of bboxes
	/// is returned.
	/// Preconditions:	The image to draw on, HOG loaded with SVM also needs to be
	///			loaded into memory, and ROI for sliding window needs to be
	///			defined.
	/// Postconditions: 	Vector of bboxes (cv::Rect) is returned
	std::vector<cv::Rect> vehicle_detect_bboxes(
		cv::Mat&,
		cv::HOGDescriptor&,
		std::vector<cv::Point>&,
		float bbox_confidence_threshold,
		float nms_threshold,
		bool include_all_bboxes = (false)
	);
	
	/// vehicle_detect_bboxes
	/// Rather than drawing the bboxes, the vector of bboxes
	/// is returned.
	/// Preconditions:	The image to draw on, HOG loaded with SVM also needs to be
	///			loaded into memory, and ROI for sliding window needs to be
	///			defined.
	/// Postconditions: 	Vector of bboxes (cv::Rect) is returned
	std::vector<cv::Rect> vehicle_detect_bboxes(
		ConfigurationParameters& config,
		cv::Mat&,
		cv::HOGDescriptor&,
		bool include_all_bboxes = (false)
	);
	
	/// respace
	/// This method takes the cropped img bboxes and 
	/// converts the bbox coordinates back into the original
	/// img coordinate space. This is because the coordinates
	/// with the bboxes are relative to the cropped ROI image.
	/// Since we only care about the top left x and y coordinates
	/// that is the only point that is converted. The rest of the bounding
	/// box is created via width and height information.
	/// Preconditions:		vector of bbox coords in space of the cropped img and ROI for the crop
	/// Postconditions:		Vector of re-scaled coordinates of bboxes relative to the original image
	std::vector<cv::Rect> respace(
		std::vector<cv::Rect>&,
		cv::Rect&
	);
	
	/// draw_bboxes
	/// Given the vector of bboxes, the bboxes are drawn on
	/// the given image.
	/// Preconditions:		The given config object must be created
	///				and provided, vector of non-max suppressed bboxes
	///				and the img to draw the boxes on must all be provided
	/// Postconditions:		img with bboxes is returned
	std::vector<cv::Rect> draw_bboxes(
		ConfigurationParameters& config,
		std::vector<cv::Rect>& bboxes,
		cv::Mat& img
	);
	
	/// Simple function that takes the number
	/// of detected cars (equivalent to the number of bboxes)
	/// and displays it on the screen.
	/// Preconditions:	img matrix to display text and bbox vector
	/// Postconditions:	img will show the number of vehicles in the current frame
	void display_num_vehicles(cv::Mat&, std::vector<cv::Rect>&);
};
#endif
