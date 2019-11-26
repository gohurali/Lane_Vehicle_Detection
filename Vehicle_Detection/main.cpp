#include "feature_extractor.h"
#include <string>
#include <iostream>

int main() {

	bool debug = true;
	std::string car_dataset_loc = "../datasets/svm_data/vehicles/vehicles/";
	std::string noncar_dataset_loc = "../datasets/svm_data/non-vehicles/non-vehicles/";
	std::string single_test = "../datasets/udacity_challenge_video/challenge_frames/201911252131_5.png";
	FeatureExtractor fe;

	if (!std::filesystem::exists("model.yaml")) {
		printf("No trained SVM Exists! Creating model...\n");
		// -- Get the dataset --
		std::pair<std::vector<cv::Mat>, std::vector<int>> dataset = fe.load_dataset(car_dataset_loc, noncar_dataset_loc, true);
		std::vector<cv::Mat> hog_ims = fe.featurize_dataset(dataset.first, false);
		std::cout << hog_ims[0].at<float>(0) << std::endl;
		//fe.show_image(hog_ims[2], 1, 1, 5000);
		std::pair<cv::Mat, cv::Mat> transformed_dataset = fe.prepare_training_data(hog_ims, dataset.second);
		std::cout << transformed_dataset.first.at<float>(0) << std::endl;
		std::cout << transformed_dataset.second.at<int>(0) << std::endl;
		fe.train_svm(transformed_dataset.first, transformed_dataset.second);
	}
	else {
		printf("Trained SVM Exists! Opening model...\n");
		cv::Mat single_im;
		if (debug) {
			single_im = cv::imread(single_test);
			cv::cvtColor(single_im, single_im, cv::COLOR_BGR2GRAY);
		}
		//fe.show_image(single_im, 1, 1, 3000);

		printf("yooooooooooooooooooooooooooooooo\n");
		std::vector<float> svm_detector = fe.get_svm_detector("model.yaml");
		printf("im backkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk\n");
		printf("size of detector is = %i", (int)svm_detector.size());
		/*for (auto i : svm_detector) {
			printf("%f , \n", i);
		}*/
		cv::HOGDescriptor hog;
		hog.winSize = cv::Size(64, 64);
		printf("Setting SVM to HOG -------------> \n");
		hog.setSVMDetector(svm_detector);
		printf("Getting ROI Coordinates ! \n");
		std::vector<cv::Point> roi = fe.detection_roi(single_im,
			0, 1,
			1, 1,
			0.5, 0.59,
			0.5, 0.59
		);
		/*printf("single im r = %i ---- c = %i\n", single_im.rows, single_im.cols);
		single_im.resize(64, 64);
		printf("single im r = %i ---- c = %i\n", single_im.rows, single_im.cols);*/

		printf("About to inference!\n");
		std::vector<cv::Point> found_locations;
		std::vector<double> location_confidence;
		single_im.convertTo(single_im, CV_8UC1);
		hog.detectROI(single_im, roi, found_locations,location_confidence);
		for (double i : location_confidence) {
			printf("%i -- \n", i);
		}

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