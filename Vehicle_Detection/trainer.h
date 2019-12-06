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

	void train_svm(
		cv::Mat&,
		cv::Mat&,
		std::string model_fname = ("model.yaml")
	);
	void train_test_svm(
		const cv::Mat&, const cv::Mat&,
		const cv::Mat&, const cv::Mat&,
		bool, std::string model_fname = ("model.yaml")
	);
	cv::Ptr<cv::ml::TrainData> train_test_split(cv::Mat&, cv::Mat&, int);

	
};

#endif // !TRAINER_H