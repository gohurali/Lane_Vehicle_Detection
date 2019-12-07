#include "inferencer.h"

/// <summary>
/// get_svm_detector
/// Opens the support vector machine serialized file
/// Preconditions:		The string location of the serialized model file
/// Postconditions:		A pair including a pointer to the SVM model and 
///						vector of float coefficients of the SVM model
/// </summary>
/// <param name="model_loc"></param>
/// <returns></returns>
// https://docs.opencv.org/3.4/d0/df8/samples_2cpp_2train_HOG_8cpp-example.html#a70
std::vector<float> Inferencer::get_svm_detector(std::string model_loc) {
	cv::Ptr<cv::ml::SVM> svm_model = cv::ml::SVM::load(model_loc);
	//cv::Ptr<cv::ml::SVM> svm_model = cv::ml::StatModel::load<cv::ml::SVM>(model_loc);
	cv::Mat support_vectors = svm_model->getSupportVectors();
	const int sv_total = support_vectors.rows;
	cv::Mat alpha;
	cv::Mat svidx;
	double rho = svm_model->getDecisionFunction(0, alpha, svidx);

	CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
	CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
		(alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
	CV_Assert(support_vectors.type() == CV_32F);
	int sz = support_vectors.cols + 1;
	std::vector<float> hog_detector;
	hog_detector.resize(sz);
	memcpy(&hog_detector[0], support_vectors.ptr(), support_vectors.cols * sizeof(hog_detector[0]));
	hog_detector[support_vectors.cols] = (float)-rho;
	return hog_detector;
}

/// <summary>
/// get_svm_detector
/// Opens the support vector machine serialized file
/// Preconditions:		The string location of the serialized model file
/// Postconditions:		A pair including a pointer to the SVM model and 
///						vector of float coefficients of the SVM model
/// </summary>
/// <param name="model_loc"></param>
/// <param name="class_num"></param>
/// <returns></returns>
std::pair<cv::Ptr<cv::ml::SVM>, std::vector<float>> Inferencer::get_svm_detector(std::string model_loc, int class_num) {
	cv::Ptr<cv::ml::SVM> svm_model = cv::ml::SVM::load(model_loc);
	int sv_dim = svm_model->getVarCount();
	//cv::Ptr<cv::ml::SVM> svm_model = cv::ml::StatModel::load<cv::ml::SVM>(model_loc);
	cv::Mat support_vectors = svm_model->getSupportVectors();
	const int sv_total = support_vectors.rows;
	cv::Mat alpha;
	cv::Mat svidx;
	alpha = cv::Mat::zeros(sv_total, sv_dim, CV_32F);
	svidx = cv::Mat::zeros(1, sv_total, CV_64F);
	cv::Mat resMat;
	double rho = svm_model->getDecisionFunction(0, alpha, svidx);
	alpha.convertTo(alpha, CV_32F);
	resMat = -1 * alpha * support_vectors;
	std::vector<float> detector;
	for (int i = 0; i < sv_dim; i++) {
		detector.push_back(resMat.at<float>(0, i));
	}
	detector.push_back(rho);
	std::pair<cv::Ptr<cv::ml::SVM>, std::vector<float>> res = { svm_model,detector };
	return res;
}


/// <summary>
/// Search for comparitable features, place bounding box, use Non max suppression to 
/// limit the number of bounding boxes per vehicle.
/// Preconditions:	The image to draw on, HOG loaded with SVM also needs to be
///					loaded into memory, and ROI for sliding window needs to be
///					defined.
///	Postconditions: img with bboxes is returned
/// </summary>
/// <param name="img"></param>
/// <param name="detector"></param>
/// <param name="roi"></param>
/// <param name="include_all_bboxes"></param>
/// <returns></returns>
cv::Mat Inferencer::vehicle_detect(cv::Mat& img, cv::HOGDescriptor& detector, std::vector<cv::Point>& roi, bool include_all_bboxes) {
	cv::Rect roi_im(
		roi[0].x, 
		roi[0].y, 
		roi[1].x - roi[0].x, 
		roi[2].y - roi[0].y
	);
	cv::Mat cropped_im = img(roi_im);
	std::vector<cv::Rect> found_locations;
	std::vector<double> confidence;
	detector.detectMultiScale(
		cropped_im, 
		found_locations, 
		confidence, 
		0.0, 
		cv::Size(8, 8), 
		cv::Size(0, 0), 
		1.2632, 
		2.0
	);
	std::vector<float> confidence_probabilities(confidence.begin(), confidence.end());
	std::vector<int> kept_boxes;
	cv::dnn::NMSBoxes(found_locations, confidence_probabilities, 0.1f, 0.1f, kept_boxes);//0.4f, 0.3f, keep_vec);

	if (include_all_bboxes) {
		for (cv::Rect r : found_locations) {
			cv::rectangle(cropped_im, r, cv::Scalar(0, 255, 0), 2);
		}
	}

	for (const int idx : kept_boxes) {
		cv::Rect curr_rect = found_locations[idx];
		cv::rectangle(cropped_im, curr_rect, cv::Scalar(0, 0, 255), 2);
	}

	cropped_im.copyTo(img(roi_im));
	return img;
}

/// <summary>
/// vehicle_detect_bboxes
/// Rather than drawing the bboxes, the vector of bboxes
/// is returned.
/// Preconditions:	The image to draw on, HOG loaded with SVM also needs to be
///					loaded into memory, and ROI for sliding window needs to be
///					defined.
///	Postconditions: Vector of bboxes (cv::Rect) is returned
/// </summary>
/// <param name="img"></param>
/// <param name="detector"></param>
/// <param name="roi"></param>
/// <param name="include_all_bboxes"></param>
/// <returns></returns>
std::vector<cv::Rect> Inferencer::vehicle_detect_bboxes(cv::Mat& img, cv::HOGDescriptor& detector, std::vector<cv::Point>& roi, bool include_all_bboxes) {
	// Define a rectangle that represents the detection ROI parameters
	cv::Rect roi_im(
		roi[0].x, 
		roi[0].y, 
		roi[1].x - roi[0].x, 
		roi[2].y - roi[0].y
	);
	// We crop that img based on the ROI rectangle location
	cv::Mat cropped_im = img(roi_im);
	std::vector<cv::Rect> found_locations;
	std::vector<double> confidence;
	// Giving default parameters that work best
	// on average
	detector.detectMultiScale(
		cropped_im,
		found_locations,
		confidence,
		0.0,
		cv::Size(8, 8),
		cv::Size(0, 0),
		1.2632,
		2.0
	);
	std::vector<float> confidence_probabilities(confidence.begin(), confidence.end());
	std::vector<int> kept_boxes;
	cv::dnn::NMSBoxes(found_locations, confidence_probabilities, 0.1f, 0.1f, kept_boxes);

	if (include_all_bboxes) {
		for (cv::Rect r : found_locations) {
			cv::rectangle(cropped_im, r, cv::Scalar(0, 255, 0), 2);
		}
	}
	std::vector<cv::Rect> filtered_boxes;
	for (const int idx : kept_boxes) {
		cv::Rect curr_rect = found_locations[idx];
		filtered_boxes.push_back(curr_rect);
	}
	return filtered_boxes;
}

/// <summary>
/// vehicle_detect_bboxes
/// Rather than drawing the bboxes, the vector of bboxes
/// is returned.
/// Preconditions:	The image to draw on, HOG loaded with SVM also needs to be
///					loaded into memory, and ROI for sliding window needs to be
///					defined.
///	Postconditions: Vector of bboxes (cv::Rect) is returned
/// </summary>
/// <param name="img"></param>
/// <param name="detector"></param>
/// <param name="roi"></param>
/// <param name="bbox_confidence_threshold"></param>
/// <param name="nms_threshold"></param>
/// <param name="include_all_bboxes"></param>
/// <returns></returns>
std::vector<cv::Rect> Inferencer::vehicle_detect_bboxes(cv::Mat& img, cv::HOGDescriptor& detector, std::vector<cv::Point>& roi, float bbox_confidence_threshold, float nms_threshold, bool include_all_bboxes) {
	// Define a rectangle that represents the detection ROI parameters
	cv::Rect roi_im(
		roi[0].x,
		roi[0].y,
		roi[1].x - roi[0].x,
		roi[2].y - roi[0].y
	);
	// We crop that img based on the ROI rectangle location
	cv::Mat cropped_im = img(roi_im);
	std::vector<cv::Rect> found_locations;
	std::vector<double> confidence;
	detector.detectMultiScale(
		cropped_im,
		found_locations,
		confidence,
		0.0,
		cv::Size(8, 8),
		cv::Size(0, 0),
		1.2632,
		2.0
	);
	std::vector<float> confidence_probabilities(confidence.begin(), confidence.end());
	std::vector<int> keep_vec;
	cv::dnn::NMSBoxes(found_locations, confidence_probabilities, bbox_confidence_threshold, nms_threshold, keep_vec);

	if (include_all_bboxes) {
		for (cv::Rect r : found_locations) {
			cv::rectangle(cropped_im, r, cv::Scalar(0, 255, 0), 2);
		}
	}
	std::vector<cv::Rect> filtered_boxes;
	for (const int idx : keep_vec) {
		cv::Rect curr_rect = found_locations[idx];
		filtered_boxes.push_back(curr_rect);
	}
	return filtered_boxes;
}

/// <summary>
/// vehicle_detect_bboxes
/// Rather than drawing the bboxes, the vector of bboxes
/// is returned.
/// Preconditions:	The image to draw on, HOG loaded with SVM also needs to be
///					loaded into memory, and ROI for sliding window needs to be
///					defined.
///	Postconditions: Vector of bboxes (cv::Rect) is returned
/// </summary>
/// <param name="config"></param>
/// <param name="img"></param>
/// <param name="detector"></param>
/// <param name="bbox_confidence_threshold"></param>
/// <param name="nms_threshold"></param>
/// <param name="include_all_bboxes"></param>
/// <returns></returns>
std::vector<cv::Rect> Inferencer::vehicle_detect_bboxes(ConfigurationParameters& config, cv::Mat& img, cv::HOGDescriptor& detector, bool include_all_bboxes) {
	// Define a rectangle that represents the detection ROI parameters
	cv::Rect roi_im(
		config.vd_roi[0].x,
		config.vd_roi[0].y,
		config.vd_roi[1].x - config.vd_roi[0].x,
		config.vd_roi[2].y - config.vd_roi[0].y
	);
	// We crop that img based on the ROI rectangle location
	cv::Mat cropped_im = img(roi_im);
	std::vector<cv::Rect> found_locations;
	std::vector<double> confidences;
	detector.detectMultiScale(
		cropped_im,
		found_locations,
		confidences,
		0.0,
		cv::Size(config.win_stride, config.win_stride),
		cv::Size(0, 0),
		config.scale_factor,
		2.0
	);
	std::vector<float> confidence_probabilities(confidences.begin(), confidences.end());
	std::vector<int> keep_vec;
	cv::dnn::NMSBoxes(
		found_locations,
		confidence_probabilities,
		config.bbox_confidence_threshold,
		config.nms_threshold,
		keep_vec
	);

	if (include_all_bboxes) {
		for (cv::Rect r : found_locations) {
			cv::rectangle(cropped_im, r, cv::Scalar(0, 255, 0), 2);
		}
	}
	std::vector<cv::Rect> filtered_boxes;
	for (const int idx : keep_vec) {
		cv::Rect curr_rect = found_locations[idx];
		filtered_boxes.push_back(curr_rect);
	}
	return filtered_boxes;
}

/// <summary>
/// respace
/// This method takes the cropped img bboxes and 
/// converts the bbox coordinates back into the original
/// img coordinate space. This is because the coordinates
/// with the bboxes are relative to the cropped ROI image.
/// Since we only care about the top left x and y coordinates
/// that is the only point that is converted. The rest of the bounding
/// box is created via width and height information.
/// Preconditions:		vector of bbox coords in space of the cropped img and ROI for the crop
/// Postconditions:		Vector of re-scaled coordinates of bboxes relative to the original image
/// </summary>
/// <param name="bboxes"></param>
/// <param name="roi"></param>
/// <returns></returns>
std::vector<cv::Rect> Inferencer::respace(std::vector<cv::Rect>& bboxes, cv::Rect& roi) {
	std::vector<cv::Rect> adjusted_bboxes;
	for (cv::Rect curr_rect : bboxes) {
		curr_rect.x += roi.x;
		curr_rect.y += roi.y;
		adjusted_bboxes.push_back(curr_rect);
	}
	return adjusted_bboxes;
}

/// <summary>
/// draw_bboxes
/// Given the vector of bboxes, the bboxes are drawn on
/// the given image.
/// Preconditions:		The given config object must be created
///						and provided, vector of non-max suppressed bboxes
///						and the img to draw the boxes on must all be provided
///	Postconditions:		img with bboxes is returned
/// </summary>
/// <param name="bboxes"></param>
/// <param name="roi"></param>
/// <returns></returns>
std::vector<cv::Rect> Inferencer::draw_bboxes(ConfigurationParameters& config, std::vector<cv::Rect>& bboxes, cv::Mat& img) {
	// Since bboxes are in the cropped image space coordinates
	// they need to be rescaled to the coordinates of the original image size
	cv::Rect roi_im(
		config.vd_roi[0].x,
		config.vd_roi[0].y,
		config.vd_roi[1].x - config.vd_roi[0].x,
		config.vd_roi[2].y - config.vd_roi[0].y
	);
	// Getting the re-scaled bbox coords for the original img size
	std::vector<cv::Rect> adjusted_bboxes = this->respace(bboxes, roi_im);
	for (const cv::Rect& box : adjusted_bboxes) {
		cv::rectangle(img, box, cv::Scalar(0, 0, 255), 2);
	}
	return adjusted_bboxes;
}

/// <summary>
/// Simple function that takes the number
/// of detected cars (equivalent to the number of bboxes)
/// and displays it on the screen.
/// Preconditions:	img matrix to display text and bbox vector
///	Postconditions:	img will show the number of vehicles in the current frame
/// </summary>
/// <param name="img"></param>
/// <param name="bboxes"></param>
void Inferencer::display_num_vehicles(cv::Mat& img, std::vector<cv::Rect>& bboxes) {
	int num_bboxes = (int)bboxes.size();
	std::string s_num_boxes = std::to_string(num_bboxes);
	cv::putText(
		img,
		cv::String("Number of Vehicles in view: " + s_num_boxes),
		cv::Point(10, img.rows - 30),
		cv::FONT_HERSHEY_DUPLEX,
		0.5,
		cv::Scalar(0, 0, 255)
	);
}