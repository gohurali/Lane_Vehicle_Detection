#ifndef CONFIG_H
#define CONFIG_H
#include <vector>
#include <string>
struct ConfigurationParameters {

	int img_width = 672;
	int img_height = 378;

	// ---------------------- Inference Data Location -------------------------
	//"../datasets/test_data/singapore_snippet1/"; //"../datasets/udacity_challenge_video/challenge_2_frames/";
	std::string test_data_loc = "../datasets/udacity_challenge_video/challenge_frames/";

	// ---------------------- Lane Detection Parameters ----------------------
	int smoothing_kernel_size = 7;
	bool remove_between_lanes = true;
	int l_threshold = 350;
	int r_threshold = 350;
	std::vector<std::pair<float, float>> inner_roi = {
									std::pair(0.4,1),
									std::pair(0.7,1),
									std::pair(0.54,0.63),
									std::pair(0.54,0.63)
	};
	std::vector<std::pair<float, float>> ld_roi = {
									std::pair(0,1),
									std::pair(1,1),
									std::pair(0.518,0.59),
									std::pair(0.518,0.59)
	};

	// ---------- HOG + Support Vector Machine Training Parameters ------
	int window_size = 64;


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
	float bbox_confidence_threshold = 0.1f;
	float nms_threshold = 0.1f;
	double scale_factor = 1.2632; // Recommended: 1.2632

};
#endif