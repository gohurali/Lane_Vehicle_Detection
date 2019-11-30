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

	// -------------- Configuration Parameters --------------
	
	bool perform_test_svm = false;
	bool seperate_svm_train = false; // SVMs for each label
	
	bool big_im = true;
	std::string model_file_name = "";
	if (big_im) {
		model_file_name = "model_big.yaml";//"test_model.yaml";
	}
	else {
		model_file_name = "model_small.yaml";
	}

	bool single_im_debug = true;
	std::string car_dataset_loc = "../datasets/svm_data/vehicles/vehicles/";
	std::string noncar_dataset_loc = "../datasets/svm_data/non-vehicles/non-vehicles/";
	std::string single_test = "../datasets/udacity_challenge_video/challenge_frames/201911271512_5.png";
	std::string dummy_img = "../datasets/x2_img.png";
	// -------------- Configuration Parameters --------------
	
	
	FeatureExtractor fe;

	if (!std::filesystem::exists(model_file_name)) {
		printf("No trained SVM Exists! Creating model...\n");
		// -- Get the dataset --

		if(seperate_svm_train){

			printf("------ Loading Images ----\n");
			std::pair<std::vector<cv::Mat>, std::vector<int>> vehicle_dataset = fe.load_dataset(
				car_dataset_loc,
				1,
				false
			);
			std::pair<std::vector<cv::Mat>, std::vector<int>> non_vehicle_dataset = fe.load_dataset(
				noncar_dataset_loc,
				-1,
				false
			);
			printf("------ Images Loaded ----\n");

			printf("------ Calculating HOGs ----\n");
			// Obtain features -- HOG
			std::vector<cv::Mat> vehicle_hog_ims = fe.featurize_dataset(vehicle_dataset.first, false);
			std::vector<cv::Mat> non_vehicle_hog_ims = fe.featurize_dataset(non_vehicle_dataset.first, false);
			printf("------ HOGs Loaded ----\n");

			printf("------ Transforming into Training Data ----\n");
			// Transform vector of matrices to matrix of matrices
			std::pair<cv::Mat, cv::Mat> v_transformed_dataset = fe.prepare_training_data(
																	vehicle_hog_ims, 
																	vehicle_dataset.second
																);
			std::pair<cv::Mat, cv::Mat> nv_transformed_dataset = fe.prepare_training_data(
																	non_vehicle_hog_ims,
																	non_vehicle_dataset.second
																);
			printf("------ Transformed into Training Data ----\n");

			printf("------ Normalizing Training Data ----\n");
			// Normalize the dataset
			cv::Mat v_norm_x_data = fe.normalize_dataset(v_transformed_dataset.first);
			cv::Mat nv_norm_x_data = fe.normalize_dataset(nv_transformed_dataset.first);
			printf("------ Done normalizing Training Data ----\n");

			if (perform_test_svm) {
				printf("-- Performing a Test on the SVM --\n");
				// Shuffle and split the data 
				cv::Ptr<cv::ml::TrainData> vehicle_dataset = fe.train_test_split(
					v_norm_x_data,
					v_transformed_dataset.second,
					25
				);
				cv::Ptr<cv::ml::TrainData> non_vehicle_dataset = fe.train_test_split(
					nv_norm_x_data,
					nv_transformed_dataset.second,
					25
				);

				printf(" -- Training/Testing Vehicle dataset SVM ---\n");
				// Train & Test SVM
				fe.train_test_svm(
					vehicle_dataset->getTrainSamples(), vehicle_dataset->getTrainResponses(),
					vehicle_dataset->getTestSamples(), vehicle_dataset->getTestResponses(),
					false
				);
				printf(" -- Done with Vehicle SVM ---\n");

				printf(" -- Training/Testing Non-Vehicle dataset SVM ---\n");
				fe.train_test_svm(
					non_vehicle_dataset->getTrainSamples(), non_vehicle_dataset->getTrainResponses(),
					non_vehicle_dataset->getTestSamples(), non_vehicle_dataset->getTestResponses(),
					false
				);
				printf(" -- Done with Non-Vehicle SVM ---\n");
			}
			else {
				printf(" -- Training Vehicle dataset SVM ---\n");
				fe.train_svm(
					v_norm_x_data, 
					v_transformed_dataset.second,
					"vehicle_model.yaml"
				);
				printf(" -- Training Non-Vehicle dataset SVM ---\n");
				fe.train_svm(
					nv_norm_x_data, 
					nv_transformed_dataset.second,
					"non_vehicle_model.yaml"
				);
			}
		}
		else {
			std::pair<std::vector<cv::Mat>, std::vector<int>> dataset = fe.load_dataset(
				car_dataset_loc,
				noncar_dataset_loc,
				false
			);
			/*std::pair<std::vector<cv::Mat>, std::vector<int>> dataset = fe.load_dataset(
				dummy_img,
				1,
				false,
				1
			);*/

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
			
	}
	else {
		printf("Trained SVM Exists! Opening model...\n");
		cv::Mat single_im;
		cv::Mat color_im;
		if (single_im_debug) {
			single_im = cv::imread(single_test);
			cv::resize(single_im, single_im, cv::Size(672,378));//(896, 504));//single_im//resize(352,624);
			//cv::GaussianBlur(single_im, single_im, cv::Size(7, 7), 5, 10);

			printf("Resized to r = %i ----- c = %i\n", single_im.rows, single_im.cols);
			color_im = single_im.clone();
			cv::cvtColor(single_im, single_im, cv::COLOR_BGR2GRAY);
		}

		//std::vector<float> svm_detector = fe.get_svm_detector(model_file_name);
		std::pair<cv::Ptr<cv::ml::SVM>, std::vector<float>> svm_items = fe.get_svm_detector(model_file_name,1);


		printf("size of detector is = %i", (int)svm_items.second.size());
		/*for (auto i : svm_detector) {
			printf("%f , \n", i);
		}*/
		cv::HOGDescriptor hog;
		hog.winSize = cv::Size(64, 64);//(40, 40)
		printf("Setting SVM to HOG -------------> \n");
		hog.setSVMDetector(svm_items.second);
		printf("Getting ROI Coordinates ! \n");
		//fe.show_image(single_im, 1, 1, 5000);
		
		//fe.show_image(single_im, 1, 1, 5000);
		/*printf("single im r = %i ---- c = %i\n", single_im.rows, single_im.cols);
		single_im.resize(64, 64);
		printf("single im r = %i ---- c = %i\n", single_im.rows, single_im.cols);*/

		printf("About to inference!\n");
		
		std::vector<double> location_confidence;
		single_im.convertTo(single_im, CV_8U);
		
		std::string mat_type = type2str(single_im.type());
		std::cout << "--------------------> " << mat_type << std::endl;
		/*std::vector<cv::Point> roi = {
										cv::Point(120,250),
										cv::Point(896,250),
										cv::Point(120,504),
										cv::Point(896,504) 
									};*/
		std::vector<cv::Point> roi = {
										cv::Point(120,175),
										cv::Point(672,175),
										cv::Point(120,300),
										cv::Point(672,300)
		};
		//std::cout << "x = " << roi[0].x << std::endl;
		//std::cout << "y = " << roi[0].y << std::endl;
		//std::cout << "width = " << roi[1].x - roi[0].x << std::endl;
		//std::cout << "height = " << roi[2].y - roi[0].y << std::endl;
		cv::Rect roi_im(roi[0].x, roi[0].y, roi[1].x - roi[0].x, roi[2].y - roi[0].y);

		cv::Mat cropped_im = color_im(roi_im);
		//std::cout << roi_im << std::endl;
		//cv::rectangle(color_im, roi_im, cv::Scalar(0, 255, 0), 2);
		//fe.show_image(color_im, 1, 1, 8000);
		cv::DetectionROI roi_obj;
		roi_obj.locations = roi;
		std::vector<cv::DetectionROI> roi_vec = { roi_obj };
		printf("----- Lets Detect ------------\n");
		cv::Size w_s = { 32, 32 };
		cv::Size w_sz = { 64, 64 };
		//std::vector<cv::Rect> found_locations = fe.sliding_window(color_im, w_s,w_sz, 1.05, svm_items.first);
		std::vector<cv::Rect> found_locations;
		std::vector<double> confidence;
		// Set the models from the training to 0, test to see if we get  any bounding boxes. 
		hog.detectMultiScale(cropped_im, found_locations,confidence, 0.0, cv::Size(10,10), cv::Size(0, 0), 1.2632, 2.0);

		std::vector<float> confidence2;
		for (const double conf : confidence) {
			confidence2.push_back(static_cast<float>(conf));
		}
		std::vector<int> keep_vec;
		cv::dnn::dnn4_v20190902::MatShape kept_boxes;
		cv::dnn::NMSBoxes(found_locations, confidence2, 0.4f, 0.3f, keep_vec);
		for (const int idx : keep_vec) {
			std::cout << idx << std::endl;
		}
		//cv::dnn::dnn4_v20190902::NMSBoxes(found_locations, confidence, 0.3f, 0.4f,kept_boxes);
		//hog.detectMultiScale(single_im, found_locations,0.0,cv::Size(16,16),cv::Size(0,0),1.11,2.0);
		//hog.detectMultiScaleROI(cropped_im, found_locations, roi_vec,2.0);
		//hog.detectROI(cropped_im,roi, found_locations,location_confidence);
		printf("----- Done Detect ------------\n");
		for (double i : location_confidence) {
			printf("%d -- \n", i);
		}
		printf("Size of found locations = %i", found_locations.size());

		for (const int idx : keep_vec) {
			cv::Rect curr_rect = found_locations[idx];
			cv::rectangle(cropped_im, curr_rect, cv::Scalar(0, 0, 255), 2);
		}

		cropped_im.copyTo(color_im(roi_im));

		/*for (cv::Rect rec : found_locations) {
			cv::rectangle(cropped_im, rec, cv::Scalar(0, 0, 255), 2);
		}*/
		cv::cvtColor(single_im, single_im, cv::COLOR_GRAY2RGB);
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