#include "feature_extractor.h"
#include <string>
#include <iostream>

std::string type2str(int type) {
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

int main() {

	bool big_im = true;
	bool perform_test_svm = true;

	std::string model_file_name = "";
	if (big_im) {
		model_file_name = "model_big.yaml";
	}
	else {
		model_file_name = "model_small.yaml";
	}

	bool debug = true;

	std::string car_dataset_loc = "../datasets/svm_data/vehicles/vehicles/";
	std::string noncar_dataset_loc = "../datasets/svm_data/non-vehicles/non-vehicles/";
	std::string single_test = "../datasets/udacity_challenge_video/challenge_frames/201911271512_5.png";
	FeatureExtractor fe;

	if (!std::filesystem::exists(model_file_name)) {
		printf("No trained SVM Exists! Creating model...\n");
		// -- Get the dataset --
		std::pair<std::vector<cv::Mat>, std::vector<int>> dataset = fe.load_dataset(
																		car_dataset_loc, 
																		noncar_dataset_loc, 
																		false
																	);
		// Obtain features -- HOG
		std::vector<cv::Mat> hog_ims = fe.featurize_dataset(dataset.first, false);
		
		// Transform vector of matrices to matrix of matrices
		std::pair<cv::Mat, cv::Mat> transformed_dataset = fe.prepare_training_data(hog_ims, dataset.second);

		// Normalize the dataset
		cv::Mat norm_x_data = fe.normalize_dataset(transformed_dataset.first);
		
		if (perform_test_svm) {
			printf("-- Performing a Test on the SVM --\n");
			// Shuffle and split the data 
			cv::Ptr<cv::ml::TrainData> dataset = fe.train_test_split(
													norm_x_data,
													transformed_dataset.second, 
													1000
												  );
			// Train & Test SVM
			fe.train_test_svm(
				dataset->getTrainSamples(), dataset->getTrainResponses(),
				dataset->getTestSamples(), dataset->getTestResponses(),
				false
			);
		}
		else {
			fe.train_svm(transformed_dataset.first, transformed_dataset.second);
		}	
	}
	else {
		printf("Trained SVM Exists! Opening model...\n");
		cv::Mat single_im;
		cv::Mat color_im;
		if (debug) {
			single_im = cv::imread(single_test);
			cv::resize(single_im, single_im, cv::Size(896, 504));//single_im//resize(352,624);
			
			printf("Resized to r = %i ----- c = %i\n", single_im.rows, single_im.cols);
			color_im = single_im.clone();
			cv::cvtColor(single_im, single_im, cv::COLOR_BGR2GRAY);
		}

		//std::vector<float> svm_detector = fe.get_svm_detector(model_file_name);
		std::vector<float> svm_detector = fe.get_svm_detector(model_file_name,1);

		printf("size of detector is = %i", (int)svm_detector.size());
		/*for (auto i : svm_detector) {
			printf("%f , \n", i);
		}*/
		cv::HOGDescriptor hog;
		hog.winSize = cv::Size(64, 64);
		printf("Setting SVM to HOG -------------> \n");
		hog.setSVMDetector(svm_detector);
		printf("Getting ROI Coordinates ! \n");
		//fe.show_image(single_im, 1, 1, 5000);
		std::vector<cv::Point> roi = fe.detection_roi(single_im,
			0, 1,
			1, 1,
			0.5, 0.49,
			1, 0.49,false
		);
		//fe.show_image(single_im, 1, 1, 5000);
		/*printf("single im r = %i ---- c = %i\n", single_im.rows, single_im.cols);
		single_im.resize(64, 64);
		printf("single im r = %i ---- c = %i\n", single_im.rows, single_im.cols);*/

		printf("About to inference!\n");
		std::vector<cv::Rect> found_locations;
		std::vector<double> location_confidence;
		single_im.convertTo(single_im, CV_8U);
		
		std::string mat_type = type2str(single_im.type());
		std::cout << "--------------------> " << mat_type << std::endl;
		
		cv::DetectionROI roi_obj;
		roi_obj.locations = roi;
		std::vector<cv::DetectionROI> roi_vec = { roi_obj };
		printf("----- Lets Detect ------------\n");
		hog.detectMultiScale(single_im, found_locations, 0.0, cv::Size(16,16), cv::Size(32, 32), 1.01, 2.0);
		//hog.detectMultiScale(single_im, found_locations,0.0,cv::Size(16,16),cv::Size(0,0),1.11,2.0);
		//hog.detectMultiScaleROI(single_im, found_locations, roi_vec);
		//hog.detectROI(single_im, roi, found_locations,location_confidence);
		printf("----- Done Detect ------------\n");
		for (double i : location_confidence) {
			printf("%d -- \n", i);
		}
		printf("Size of found locations = %i", found_locations.size());

		for (cv::Rect rec : found_locations) {
			cv::rectangle(color_im, rec, cv::Scalar(0, 0, 255), 2);
		}
		fe.show_image(color_im, 1, 1, 5000);
		cv::imwrite("current_detections.png", color_im);
	}
	return 0;
}

//int main() {
//	bool debug = false;
//	
//	std::string path = "../datasets/udacity_challenge_video/challenge_frames/";
//	std::string single_im_path = "../datasets/udacity_challenge_video/challenge_frames/201911271512_2.png";
//	FeatureExtractor fe;
//
//	if (debug) {
//		cv::Mat input = cv::imread(single_im_path);
//		cv::Mat output = fe.lane_detect(input);
//		fe.show_image(output, 1, 1, 20000);
//	}
//	else {
//		int num = 0;
//		for (const auto& entry : std::filesystem::directory_iterator(path)) {
//			//std::cout << entry.path() << std::endl;
//			std::string current_im_loc = entry.path().string();
//			cv::Mat input = cv::imread(current_im_loc);
//			cv::Mat output = fe.lane_detect(input);
//			std::string num_name = std::to_string(num);
//			cv::imwrite("outputs/" + num_name + ".png", output);
//			num++;
//		}
//	}
//	return 0;
//}