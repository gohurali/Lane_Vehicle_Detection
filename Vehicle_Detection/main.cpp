/// Main
/// By Gohur Ali, Umair Qureshi, and Will Thomas
/// Driver file for vehicle and lane detection. 
/// Vehicle Detection is completed by training an SVM model from a 
/// dataset of car images. 
/// Lane detection utilizes the Hough Transform to identify lane 
/// lines
/// Vehicles are identified with bounding boxes.
/// Lane lines are highlighted and the region with in is shaded.
/// Pre: Road image and dataset of cars
/// Post: An image with cars and lane lines highlighted, as well as a 
/// car count per frame displayed on the bottom of the image
#include "feature_extractor.h"
#include "trainer.h"
#include "inferencer.h"

/// lane_vehicle_detect
/// Preconditions:		Config, FeatureExtractor, and Inferencer object should be passed in
/// Postconditions:		Lane and Vehicle detection images will be placed in the output location
///						as specified inthe config.h file
void lane_vehicle_detect(ConfigurationParameters& config, FeatureExtractor& fe,Inferencer& inf);

/// train_model
/// Preconditions:		Provide config object, FeatureExtractor, and Trainer object
/// Postconditions:		Serialized SVM will be placed in the current directory
void train_model(ConfigurationParameters& config, FeatureExtractor& fe, Trainer& trainer);

/// Main function
/// Precondtions:	Ensure that the datasets are present in the specified locations in 
///					the config.h file
/// Postconditions:	Detection frame images will be placed on in the output directory
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

/// train_model
/// Preconditions:		Provide config object, FeatureExtractor, and Trainer object
/// Postconditions:		Serialized SVM will be placed in the current directory
void train_model(ConfigurationParameters& config, FeatureExtractor& fe, Trainer& trainer) {
	std::pair<std::vector<cv::Mat>, std::vector<int>> dataset = fe.load_dataset(
		config.car_dataset_loc,
		config.noncar_dataset_loc,
		false
	);

	// Obtain features -- HOG
	std::vector<cv::Mat> hog_ims = fe.featurize_dataset(config,dataset.first, false);

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
			config.test_set_size
		);
		// Train & Test SVM
		trainer.train_test_svm(
			dataset->getTrainSamples(), dataset->getTrainResponses(),
			dataset->getTestSamples(), dataset->getTestResponses(),
			config.save_tested_model
		);
	}
	else {
		trainer.train_svm(transformed_dataset.first, transformed_dataset.second);
	}
}

/// lane_vehicle_detect
/// Preconditions:		Config, FeatureExtractor, and Inferencer object should be passed in
/// Postconditions:		Lane and Vehicle detection images will be placed in the output location
///						as specified inthe config.h file
void lane_vehicle_detect(ConfigurationParameters& config, FeatureExtractor& fe, Inferencer& inf) {
	printf("---- Opening SVM Model ---\n");
	std::pair<cv::Ptr<cv::ml::SVM>, std::vector<float>> svm_items = inf.open_model(config.model_name);
	cv::HOGDescriptor hog;
	hog.winSize = cv::Size(config.window_size, config.window_size);
	hog.setSVMDetector(svm_items.second);
	printf("---- Model opened ----\n");

	// Iterating through the location where the video image frames are located
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
		cv::imwrite(config.output_loc + name_num + ".png", ld_out);
	}
	printf("[LOG]: Dection done!\n");

}
