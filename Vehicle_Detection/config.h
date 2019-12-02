#ifndef CONFIG_H
#define CONFIG_H
#include <vector>
#include <string>
struct ConfigurationParameters {
	//"../datasets/test_data/singapore_snippet1/"; //"../datasets/udacity_challenge_video/challenge_2_frames/";
	std::string test_data_loc = "../datasets/udacity_challenge_video/challenge_2_frames/";

	// ---------------------- Lane Detection Parameters ----------------------
	std::vector<std::pair<float, float>> inner_roi = {
									std::pair(0.4,1),
									std::pair(0.7,1),
									std::pair(0.54,0.63),
									std::pair(0.54,0.63)
	};
	std::vector<std::pair<float, float>> ld_roi = {
									std::pair(0,1),
									std::pair(1,1),
									std::pair(0.518,0.61),
									std::pair(0.518,0.61)
	};

	// ---------------------- Vehicle Detection Parameters ----------------------
	std::string model_name = "model_big.yaml";
	std::string car_dataset_loc = "../datasets/svm_data/vehicles/vehicles/";
	std::string noncar_dataset_loc = "../datasets/svm_data/non-vehicles/non-vehicles/";

	std::vector<cv::Point> vd_roi = {
									cv::Point(120,175),
									cv::Point(672,175),
									cv::Point(120,300),
									cv::Point(672,300)
	};
	float bbox_confidence_threshold = 0.3f;
	float nms_threshold = 0.1f;

};
#endif