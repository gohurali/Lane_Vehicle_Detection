#include "feature_extractor.h"
#include <filesystem>
#include <string>
#include <iostream>

int main() {
	bool debug = false;
	
	std::string path = "../datasets/udacity_challenge_video/challenge_frames/";
	//std::string path = "../datasets/udacity_challenge_video/challenge_frames/20191111146_1.png";
	FeatureExtractor fe;
	int num = 0;
	for (const auto& entry : std::filesystem::directory_iterator(path)) {
		//std::cout << entry.path() << std::endl;
		std::string current_im_loc = entry.path().string();
		cv::Mat input = cv::imread(current_im_loc);
		cv::Mat output = fe.lane_detect(input);
		std::string num_name = std::to_string(num);
		cv::imwrite("outputs/" + num_name + ".png",output);
		num++;
	}

	//cv::Mat input = cv::imread(path);
	//cv::Mat output = fe.lane_detect(input);
	//fe.show_image(output, 1, 1, 20000);


	return 0;
}