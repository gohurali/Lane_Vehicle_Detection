#include "feature_extractor.h"
#include <string>
#include <iostream>

int main() {
	std::string car_dataset_loc = "../datasets/svm_data/vehicles/vehicles/";
	std::string noncar_dataset_loc = "../datasets/svm_data/non-vehicles/non-vehicles/";
	FeatureExtractor fe;

	if (!std::filesystem::exists("model.yaml")) {
		printf("No trained SVM Exists! Creating model...\n");
		// -- Get the dataset --
		std::pair<std::vector<cv::Mat>, std::vector<int>> dataset = fe.load_dataset(car_dataset_loc, noncar_dataset_loc, false);
		std::vector<cv::Mat> hog_ims = fe.featurize_dataset(dataset.first, false);
		std::pair<cv::Mat, cv::Mat> transformed_dataset = fe.prepare_training_data(hog_ims, dataset.second);
		fe.train_svm(transformed_dataset.first, transformed_dataset.second);
	}
	else {
		printf("Trained SVM Exists! Opening model...\n");
	}
	
	
	return 0;
}

//int main() {
//	bool debug = false;
//	
//	std::string path = "../datasets/udacity_challenge_video/challenge_frames/";
//	//std::string path = "../datasets/udacity_challenge_video/challenge_frames/20191111146_1.png";
//	FeatureExtractor fe;
//	int num = 0;
//	for (const auto& entry : std::filesystem::directory_iterator(path)) {
//		//std::cout << entry.path() << std::endl;
//		std::string current_im_loc = entry.path().string();
//		cv::Mat input = cv::imread(current_im_loc);
//		cv::Mat output = fe.lane_detect(input);
//		std::string num_name = std::to_string(num);
//		cv::imwrite("outputs/" + num_name + ".png",output);
//		num++;
//	}
//
//	//cv::Mat input = cv::imread(path);
//	//cv::Mat output = fe.lane_detect(input);
//	//fe.show_image(output, 1, 1, 20000);
//
//
//	return 0;
//}