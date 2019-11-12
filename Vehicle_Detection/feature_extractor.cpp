#include "feature_extractor.h"

cv::Mat FeatureExtractor::mask_color(cv::Mat& img, std::vector<int>& lower_b, std::vector<int>& upper_b) {
	cv::Mat lower_bound(lower_b);
	cv::Mat upper_bound(upper_b);
	cv::Mat mask;
	cv::inRange(img, lower_bound, upper_bound,mask);
	return mask;
}

cv::Mat FeatureExtractor::combine_mask(cv::Mat& mask1, cv::Mat& mask2) {
	cv::Mat combined;
	cv::bitwise_or(mask1, mask2, combined);
	return combined;
}

cv::Mat FeatureExtractor::propose_roi(cv::Mat& input, double top_l1, double top_l2,
														double top_r1, double top_r2,
														double bottom_l1, double bottom_l2 ,
														double bottom_r1, double bottom_r2) {
	input.convertTo(input, CV_64F);

	std::pair<int, int> top_left = { input.cols * top_l1 , input.rows * top_l2 };
	std::pair<int,int> top_right = { input.cols * top_r1 ,input.rows * top_r2 };
	std::pair<int, int> bottom_left = { input.cols * bottom_l1,input.rows * bottom_l2 };
	std::pair<int, int> bottom_right = { input.cols * bottom_r1,input.rows * bottom_r2 };

	cv::Point top_left_pt;
	top_left_pt.x = input.cols * top_l1;
	top_left_pt.y = input.rows * top_l2;

	cv::Point top_right_pt;
	top_right_pt.x = input.cols * top_r1;
	top_right_pt.y = input.rows * top_r2;

	cv::Point bottom_left_pt;
	bottom_left_pt.x = input.cols * bottom_l1;
	bottom_left_pt.y = input.rows * bottom_l2;

	cv::Point bottom_right_pt;
	bottom_right_pt.x = input.cols * bottom_r1;
	bottom_right_pt.y = input.rows * bottom_r2;

	cv::Point corners[1][4];
	corners[0][0] = bottom_left_pt;
	corners[0][1] = top_left_pt;
	corners[0][2] = top_right_pt;
	corners[0][3] = bottom_right_pt;

	const cv::Point* corner_list[1] = { corners[0] };
	int num_points = 4;
	int num_polygons = 1;
	
	cv::Mat mask = cv::Mat::zeros(cv::Size(input.cols,input.rows), CV_64F);
	//cv::line(input, bottom_left_pt, top_right_pt, 255);
	//cv::line(input, bottom_right_pt, top_left_pt, 255);

	cv::fillPoly(mask,corner_list,&num_points,num_polygons,cv::Scalar(255));
	cv::Mat output(cv::Size(mask.cols,mask.rows), CV_64F);
	cv::bitwise_and(input,mask,output);
	return output;
}

cv::Mat FeatureExtractor::get_lanes(cv::Mat& input,cv::Mat& output) {
	input.convertTo(input, CV_8UC1);
	std::vector<cv::Vec4i> lines;;
	//lines.convertTo(lines, CV_8UC1);
	int rho = 1;
	double theta = M_PI/180;
	int threshold = 20;
	int minLineLength = 20;
	int maxLineGap = 300;
	cv::HoughLinesP(input,lines, rho, theta,threshold,minLineLength,maxLineGap);

	for (int i = 0; i < lines.size(); i++) {
		cv::Vec4i pts = lines[i];
		cv::line(output, cv::Point(pts[0], pts[1]), cv::Point(pts[2], pts[3]), cv::Scalar(0, 0, 255));
	}


	return output;
}

cv::Mat FeatureExtractor::lane_detect(cv::Mat& input_frame) {
	// Convert to HLS color space
	cv::Mat hls_im;
	cv::cvtColor(input_frame, hls_im, cv::COLOR_BGR2HLS);

	// Get RGB img
	cv::Mat rgb_im;
	cv::cvtColor(hls_im, rgb_im, cv::COLOR_HLS2RGB);

	// Get yellow lanes
	std::vector<int> y_lower_b = { 10, 0, 100 };
	std::vector<int> y_upper_b = { 40, 255, 255 };
	cv::Mat yellow_mask = this->mask_color(hls_im, y_lower_b, y_upper_b);

	// Get white lanes
	std::vector<int> w_lower_b = { 0, 200, 0 };
	std::vector<int> w_upper_b = { 200, 255, 255 };
	cv::Mat white_mask = this->mask_color(hls_im, w_lower_b, w_upper_b);

	// Combine
	cv::Mat combined = this->combine_mask(yellow_mask, white_mask);

	//Blur
	cv::Mat blurred;
	cv::GaussianBlur(combined, blurred, { 7,7 }, 0);

	// Edge detect
	cv::Mat edges;
	cv::Canny(blurred, edges, 100, 190);

	// From the edges, remove unnecessary edges
	// propose a region of interest
	cv::Mat roi_im = this->propose_roi(edges,
		0, 1,
		1, 1,
		0.5, 0.59,
		0.5, 0.59
	);

	rgb_im = this->get_lanes(roi_im, rgb_im);
	return rgb_im;
}

void FeatureExtractor::show_image(cv::Mat& img, int x_window, int y_window, int delay) {
	cv::namedWindow("output", cv::WINDOW_NORMAL);
	cv::resizeWindow("output", img.cols / x_window, img.rows / y_window);
	cv::imshow("output", img);
	cv::waitKey(delay);
}