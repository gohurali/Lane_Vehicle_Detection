/// Trainer 
/// By Gohur, Umair, Will
/// Extracts features from dataset. Trains a support vectore 
/// machine model for image classifcation and car detection
/// Pre: Dataset of cars
/// Post: SVM model of car features
#ifndef TRAINER_H
#define TRAINER_H
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

class Trainer {
public:

	/// Train support vector machine model for classification
	/// on x and y dataset
	/// Preconditions:		x and y data must be formatted as matrices and specified model_name
	/// Postconditions:		model is saved in the current project directory
	void train_svm(
		cv::Mat&,
		cv::Mat&,
		std::string model_fname = ("model.yaml")
	);
	
	/// train_test_svm
	/// Debug method where the split dataset is used
	/// model is then tested with the test set to see
	/// the model performance
	/// Preconditions:		The dataset must be split into train and test sets, additionally the user
	///				can specify if would like to serialize the model
	/// Postconditions:		Returns the model's performance and serializes the model if specified by user
	void train_test_svm(
		const cv::Mat&, const cv::Mat&,
		const cv::Mat&, const cv::Mat&,
		bool, std::string model_fname = ("model.yaml")
	);
	
	/// train_test_split
	/// Preconditions:		x_data and y_data and a defined test set size are needed to be passed in
	/// Postconditions:		Pointer to TrainData dataset is returned to be able to get training and test data
	cv::Ptr<cv::ml::TrainData> train_test_split(cv::Mat&, cv::Mat&, int);
};
#endif // !TRAINER_H