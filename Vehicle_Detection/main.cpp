#include "feature_extractor.h"
#include "trainer.h"
#include "inferencer.h"

void lane_vehicle_detect(ConfigurationParameters& config, FeatureExtractor& fe,Inferencer& inf);
void train_model(ConfigurationParameters& config, FeatureExtractor& fe, Trainer& trainer);

int main() {
	ConfigurationParameters config;
	FeatureExtractor fe;
	Trainer trainer;
	Inferencer inf;

	if (!std::filesystem::exists(config.model_name)) {
		printf("---- Training model ----\n");
		train_model(config, fe,trainer);
	}
	else {
		printf("---- Inference on Model ----- \n");
		lane_vehicle_detect(config, fe,inf);
	}

	
	return 0;
}

void train_model(ConfigurationParameters& config, FeatureExtractor& fe, Trainer& trainer) {
	std::pair<std::vector<cv::Mat>, std::vector<int>> dataset = fe.load_dataset(
		config.car_dataset_loc,
		config.noncar_dataset_loc,
		false
	);

	// Obtain features -- HOG
	std::vector<cv::Mat> hog_ims = fe.featurize_dataset(dataset.first, false);

	// Transform vector of matrices to matrix of matrices
	std::pair<cv::Mat, cv::Mat> transformed_dataset = fe.prepare_training_data(hog_ims, dataset.second);

	// Normalize the dataset
	cv::Mat norm_x_data = fe.normalize_dataset(transformed_dataset.first);

	if (config.perform_test_svm) {
		printf("-- Performing a Test on the SVM --\n");
		// Shuffle and split the data 
		cv::Ptr<cv::ml::TrainData> dataset = trainer.train_test_split(
			norm_x_data,
			transformed_dataset.second,
			1000
		);
		// Train & Test SVM
		trainer.train_test_svm(
			dataset->getTrainSamples(), dataset->getTrainResponses(),
			dataset->getTestSamples(), dataset->getTestResponses(),
			false
		);
	}
	else {
		trainer.train_svm(transformed_dataset.first, transformed_dataset.second);
	}
}

void lane_vehicle_detect(ConfigurationParameters& config, FeatureExtractor& fe, Inferencer& inf) {
	printf("---- Opening SVM Model ---\n");
	std::pair<cv::Ptr<cv::ml::SVM>, std::vector<float>> svm_items = inf.get_svm_detector(config.model_name, 1);
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
		std::vector<cv::Rect> bboxes = inf.vehicle_detect_bboxes(
			config,
			img_frame,
			hog,
			false
		);

		std::vector<cv::Rect> adjusted_bboxes = inf.draw_bboxes(config, bboxes, ld_out);

		// Count number of detections
		inf.display_num_vehicles(ld_out, adjusted_bboxes);
		cv::imwrite("ld_vd_imgs/" + name_num + ".png", ld_out);
	}
	printf("[LOG]: Dection done!\n");

}