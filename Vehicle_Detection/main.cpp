#include "feature_extractor.h"

int main() {
	ConfigurationParameters config;
	FeatureExtractor fe;

	printf("---- Opening SVM Model ---\n");
	std::pair<cv::Ptr<cv::ml::SVM>, std::vector<float>> svm_items = fe.get_svm_detector(config.model_name, 1);
	cv::HOGDescriptor hog;
	hog.winSize = cv::Size(config.window_size, config.window_size);
	hog.setSVMDetector(svm_items.second);
	printf("---- Model opened ----\n");
	
	for (const auto& entry : std::filesystem::directory_iterator(config.test_data_loc)) {
		std::string current_im_loc = entry.path().string();
		std::cout << current_im_loc << std::endl;
		std::string name_num = fe.get_name_num(current_im_loc);

		cv::Mat img_frame = cv::imread(current_im_loc);
		cv::resize(img_frame, img_frame, cv::Size(config.img_width, config.img_height));

		// Find the lanes on the road
		cv::Mat ld_out = fe.lane_detect(config, img_frame);
		
		// Find localized bboxes on img
		std::vector<cv::Rect> bboxes = fe.vehicle_detect_bboxes(
			config,
			img_frame, 
			hog,
			false
		);

		std::vector<cv::Rect> adjusted_bboxes = fe.draw_bboxes(config, bboxes, ld_out);

		// Count number of detections
		fe.display_num_vehicles(ld_out, adjusted_bboxes);
		cv::imwrite("ld_vd_imgs/" + name_num + ".png", ld_out);
	}
	printf("[LOG]: Dection done!\n");
	return 0;
}



int vehicle_detection() {

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

	bool single_im_debug = false;
	std::string car_dataset_loc = "../datasets/svm_data/vehicles/vehicles/";
	std::string noncar_dataset_loc = "../datasets/svm_data/non-vehicles/non-vehicles/";
	std::string single_test = "../datasets/udacity_challenge_video/challenge_frames/201911271512_3.png";
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
		if (single_im_debug) {
			cv::Mat single_im;
			cv::Mat color_im;
			single_im = cv::imread(single_test);
			cv::resize(single_im, single_im, cv::Size(672,378));//(896, 504));//single_im//resize(352,624);
			

			printf("Resized to r = %i ----- c = %i\n", single_im.rows, single_im.cols);
			color_im = single_im.clone();
			cv::cvtColor(single_im, single_im, cv::COLOR_BGR2GRAY);

			std::pair<cv::Ptr<cv::ml::SVM>, std::vector<float>> svm_items = fe.get_svm_detector(model_file_name, 1);
			cv::HOGDescriptor hog;
			hog.winSize = cv::Size(64, 64);
			hog.setSVMDetector(svm_items.second);

			std::vector<cv::Point> roi = {
											cv::Point(120,175),
											cv::Point(672,175),
											cv::Point(120,300),
											cv::Point(672,300)
			};
			color_im = fe.vehicle_detect(color_im, hog, roi,false);

			fe.show_image(color_im, 1, 1, 5000);
			cv::imwrite("current_detections.png", color_im);
		}
		else {

			std::pair<cv::Ptr<cv::ml::SVM>, std::vector<float>> svm_items = fe.get_svm_detector(model_file_name, 1);
			cv::HOGDescriptor hog;
			hog.winSize = cv::Size(64, 64);
			hog.setSVMDetector(svm_items.second);

			std::vector<cv::Point> roi = {
											cv::Point(120,175),
											cv::Point(672,175),
											cv::Point(120,300),
											cv::Point(672,300)
										 };

			std::string path = "../datasets/udacity_challenge_video/challenge_frames/";
			int count = 0;
			for (const auto& entry : std::filesystem::directory_iterator(path)) {
				std::string current_im_loc = entry.path().string();
				cv::Mat img = cv::imread(current_im_loc);
				cv::resize(img, img, cv::Size(672, 378));
				img = fe.vehicle_detect(img, hog, roi);
				std::string num_name = std::to_string(count);
				cv::imwrite("vehicle_detection_imgs/" + num_name + ".png", img);
				count++;
			}
		}
	}
	return 0;
}

int lane_detection() {
	bool debug = false;
	
	std::string path = "../datasets/udacity_challenge_video/challenge_frames/";
	std::string single_im_path = "../datasets/udacity_challenge_video/challenge_frames/201911271512_2.png";
	FeatureExtractor fe;

	if (debug) {
		cv::Mat input = cv::imread(single_im_path);
		cv::Mat output = fe.lane_detect(input);
		fe.show_image(output, 1, 1, 20000);
	}
	else {
		int num = 0;
		for (const auto& entry : std::filesystem::directory_iterator(path)) {
			//std::cout << entry.path() << std::endl;
			std::string current_im_loc = entry.path().string();
			cv::Mat input = cv::imread(current_im_loc);
			cv::Mat output = fe.lane_detect(input);
			std::string num_name = std::to_string(num);
			cv::imwrite("outputs/" + num_name + ".png", output);
			num++;
		}
	}
	return 0;
}