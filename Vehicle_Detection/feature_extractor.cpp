/// FeatureExtractor 
#include "feature_extractor.h"

/// Creates a mask given lower and upper bound limitations
/// Pre:
/// Mat img must be an image
/// vector lower_b must be a vector of lower bounds
/// vector upper_b must be a vector of upper bounds
/// Post:
/// Returns the mask that is derived from the input image and the specified bounds
cv::Mat FeatureExtractor::mask_color(cv::Mat& img, std::vector<int>& lower_b, std::vector<int>& upper_b) {
	cv::Mat lower_bound(lower_b);
	cv::Mat upper_bound(upper_b);
	cv::Mat mask;
	cv::inRange(img, lower_bound, upper_bound,mask);
	return mask;
}

/// combine_mask  creates a combined bit-wise disjunction of mask1 and mask2
/// Pre:
/// Mask1 a mat that is a mask
/// Mask2 a mat that is a mask
/// Post:
/// combined is the bit-wise disjunction of mask1 and 2.
cv::Mat FeatureExtractor::combine_mask(cv::Mat& mask1, cv::Mat& mask2) {
	cv::Mat combined;

	// Finds the per-element bit-wise disjunction of both masks and stores into
	// combined Mat
	cv::bitwise_or(mask1, mask2, combined);
	return combined;
}

cv::Mat FeatureExtractor::remove_middle_polygons(cv::Mat& edge_im, cv::Mat& mask) {
	for (int row = 0; row < edge_im.rows; row++) {
		for (int col = 0; col < edge_im.cols; col++) {
			int edge_curr_pxl = edge_im.at<uchar>(row, col);
			int mask_curr_pxl = mask.at<uchar>(row, col);
			if (edge_curr_pxl == 255 && mask_curr_pxl == 255) {
				edge_im.at<uchar>(row, col) = 0;
			}
		}
	}
	return edge_im;
}

cv::Mat FeatureExtractor::create_inner_cover_mask(cv::Mat& input, std::vector<std::pair<float, float>>& roi, bool debug) {

	cv::Point top_left_pt;
	top_left_pt.x = input.cols * roi[0].first;
	top_left_pt.y = input.rows * roi[0].second;

	cv::Point top_right_pt;
	top_right_pt.x = input.cols * roi[1].first;
	top_right_pt.y = input.rows * roi[1].second;

	cv::Point bottom_left_pt;
	bottom_left_pt.x = input.cols * roi[2].first;
	bottom_left_pt.y = input.rows * roi[2].second;

	cv::Point bottom_right_pt;
	bottom_right_pt.x = input.cols * roi[3].first;
	bottom_right_pt.y = input.rows * roi[3].second;

	cv::Point corners[1][4];
	corners[0][0] = bottom_left_pt;
	corners[0][1] = top_left_pt;
	corners[0][2] = top_right_pt;
	corners[0][3] = bottom_right_pt;

	const cv::Point* corner_list[1] = { corners[0] };
	int num_points = 4;
	int num_polygons = 1;

	cv::Mat mask = cv::Mat::zeros(cv::Size(input.cols, input.rows), CV_64F);

	// Debug Draw Methods
	if (debug) {
		cv::line(input, bottom_left_pt, top_right_pt, 255);
		cv::line(input, bottom_right_pt, top_left_pt, 255);
		this->show_image(input, 1, 1, 5000);
	}
	cv::fillPoly(mask, corner_list, &num_points, num_polygons, cv::Scalar(255));
	return mask;
}

/// Propose Region of Interest
cv::Mat FeatureExtractor::propose_roi(cv::Mat& input, std::vector<std::pair<float, float>>& roi,bool debug) {
	input.convertTo(input, CV_64F);

	cv::Point top_left_pt;
	top_left_pt.x = input.cols * roi[0].first;
	top_left_pt.y = input.rows * roi[0].second;

	cv::Point top_right_pt;
	top_right_pt.x = input.cols * roi[1].first;
	top_right_pt.y = input.rows * roi[1].second;

	cv::Point bottom_left_pt;
	bottom_left_pt.x = input.cols * roi[2].first;
	bottom_left_pt.y = input.rows * roi[2].second;

	cv::Point bottom_right_pt;
	bottom_right_pt.x = input.cols * roi[3].first;
	bottom_right_pt.y = input.rows * roi[3].second;

	cv::Point corners[1][4];
	corners[0][0] = bottom_left_pt;
	corners[0][1] = top_left_pt;
	corners[0][2] = top_right_pt;
	corners[0][3] = bottom_right_pt;

	const cv::Point* corner_list[1] = { corners[0] };
	int num_points = 4;
	int num_polygons = 1;

	cv::Mat mask = cv::Mat::zeros(cv::Size(input.cols, input.rows), CV_64F);

	// Debug Draw Methods
	if (debug) {
		cv::line(input, bottom_left_pt, top_right_pt, 255);
		cv::line(input, bottom_right_pt, top_left_pt, 255);
		this->show_image(input, 1, 1, 5000);
	}

	cv::fillPoly(mask, corner_list, &num_points, num_polygons, cv::Scalar(255));
	cv::Mat output(cv::Size(mask.cols, mask.rows), CV_64F);
	cv::bitwise_and(input, mask, output);
	return output;
}

cv::Mat FeatureExtractor::propose_roi(cv::Mat& input, double top_l1, double top_l2,
													  double top_r1, double top_r2,
													  double bottom_l1, double bottom_l2,
													  double bottom_r1, double bottom_r2, bool debug) {
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

	// Debug Draw Methods
	if (debug) {
		cv::line(input, bottom_left_pt, top_right_pt, 255);
		cv::line(input, bottom_right_pt, top_left_pt, 255);
		this->show_image(input, 1, 1, 5000);
	}

	cv::fillPoly(mask,corner_list,&num_points,num_polygons,cv::Scalar(255));
	cv::Mat output(cv::Size(mask.cols,mask.rows), CV_64F);
	cv::bitwise_and(input,mask,output);
	return output;
}

/// Get Lanes 
cv::Mat FeatureExtractor::get_lanes(cv::Mat& input, cv::Mat& output, int l_threshold, int r_threshold) {
	input.convertTo(input, CV_8UC1);
	std::vector<cv::Vec4i> lines;
	//lines.convertTo(lines, CV_8UC1);
	int rho = 1;
	double theta = M_PI / 180;
	int threshold = 20;
	int minLineLength = 20;
	int maxLineGap = 300;
	cv::HoughLinesP(input, lines, rho, theta, threshold, minLineLength, maxLineGap);
	if (lines.size() == 0) {
		return output;
	}

	std::vector<cv::Vec4i> hough_lines;
	for (int i = 0; i < lines.size(); i++) {
		cv::Vec4i pts = lines[i];
		cv::Point x1y1(pts[0], pts[1]);
		cv::Point x2y2(pts[2], pts[3]);
		hough_lines.push_back(pts);
		//cv::line(output, x1y1,x2y2, cv::Scalar(0, 0, 255));
	}
	//this->show_image(output, 1, 1, 5000);

	// Look for highest point
	cv::Vec4i top_line = this->find_highest_point(hough_lines, input.cols / 2, true);
	cv::line(
		output,
		cv::Point(top_line[0], top_line[1]),
		cv::Point(top_line[2], top_line[3]),
		cv::Scalar(0, 255, 0),
		2
	);

	// Look for the lowest point
	cv::Vec4i min_points = this->find_lowest_point(hough_lines, input.cols / 2, true);

	// Extrapolate the right lane
	cv::Vec4i current_right_lane = { min_points[2],min_points[3],top_line[2],top_line[3] };
	//std::cout << " Running \n";
	//std::cout << current_right_lane << std::endl;
	cv::Point adjusted_right_min = this->extrapolate_line(current_right_lane, min_points[1]);
	//std::cout << adjusted_right_min << std::endl;

	cv::line(
		output,
		cv::Point(min_points[0], min_points[1]),
		cv::Point(top_line[0], top_line[1]),
		cv::Scalar(0, 255, 0),
		2
	);

	cv::line(
		output,
		cv::Point(adjusted_right_min.x, adjusted_right_min.y),
		cv::Point(top_line[2], top_line[3]),
		cv::Scalar(0, 255, 0),
		2
	);

	cv::Point corners[1][4];
	corners[0][0] = cv::Point(min_points[0], min_points[1]);
	corners[0][1] = cv::Point(adjusted_right_min.x, adjusted_right_min.y);
	corners[0][2] = cv::Point(top_line[2], top_line[3]);
	corners[0][3] = cv::Point(top_line[0], top_line[1]);

	const cv::Point* corner_list[1] = { corners[0] };
	int num_points = 4;
	int num_polygons = 1;
	cv::Mat overlay;
	output.copyTo(overlay);
	cv::fillPoly(overlay, corner_list, &num_points, num_polygons, cv::Scalar(0, 255, 0));
	double alpha = 0.3;

	cv::addWeighted(overlay, alpha, output, 1 - alpha, 0, output);
	return output;
}


cv::Mat FeatureExtractor::get_lanes(cv::Mat& input,cv::Mat& output) {
	input.convertTo(input, CV_8UC1);
	std::vector<cv::Vec4i> lines;
	//lines.convertTo(lines, CV_8UC1);
	int rho = 1;
	double theta = M_PI/180;
	int threshold = 20;
	int minLineLength = 20;
	int maxLineGap = 300;
	cv::HoughLinesP(input,lines, rho, theta,threshold,minLineLength,maxLineGap);
	if (lines.size() == 0) {
		return output;
	}

	std::vector<cv::Vec4i> hough_lines;
	for (int i = 0; i < lines.size(); i++) {
		cv::Vec4i pts = lines[i];
		cv::Point x1y1(pts[0], pts[1]);
		cv::Point x2y2(pts[2], pts[3]);
		hough_lines.push_back(pts);
		//cv::line(output, x1y1,x2y2, cv::Scalar(0, 0, 255));
	}
	//this->show_image(output, 1, 1, 5000);

	// Look for highest point
	cv::Vec4i top_line = this->find_highest_point(hough_lines,input.cols/2, true);
	cv::line(
		output, 
		cv::Point(top_line[0], top_line[1]), 
		cv::Point(top_line[2], top_line[3]), 
		cv::Scalar(0, 255, 0),
		2
	);

	// Look for the lowest point
	cv::Vec4i min_points = this->find_lowest_point(hough_lines, input.cols / 2, true);

	// Extrapolate the right lane
	cv::Vec4i current_right_lane = { min_points[2],min_points[3],top_line[2],top_line[3]};
	//std::cout << " Running \n";
	//std::cout << current_right_lane << std::endl;
	cv::Point adjusted_right_min = this->extrapolate_line(current_right_lane, min_points[1]);
	//std::cout << adjusted_right_min << std::endl;

	cv::line(
		output, 
		cv::Point(min_points[0], min_points[1]), 
		cv::Point(top_line[0], top_line[1]), 
		cv::Scalar(0, 255, 0),
		2
	);

	cv::line(
		output, 
		cv::Point(adjusted_right_min.x, adjusted_right_min.y), 
		cv::Point(top_line[2], top_line[3]), 
		cv::Scalar(0, 255, 0),
		2
	);

	cv::Point corners[1][4];
	corners[0][0] = cv::Point(min_points[0], min_points[1]);
	corners[0][1] = cv::Point(adjusted_right_min.x, adjusted_right_min.y);
	corners[0][2] = cv::Point(top_line[2], top_line[3]); 
	corners[0][3] = cv::Point(top_line[0], top_line[1]);

	const cv::Point* corner_list[1] = { corners[0] };
	int num_points = 4;
	int num_polygons = 1;
	cv::Mat overlay;
	output.copyTo(overlay);
	cv::fillPoly(overlay, corner_list, &num_points, num_polygons, cv::Scalar(0,255,0));
	double alpha = 0.3;
	
	cv::addWeighted(overlay, alpha, output, 1 - alpha, 0, output);
	return output;
}

cv::Point FeatureExtractor::extrapolate_line(cv::Vec4i& line, int y_pt) {
	cv::Point right_bottom(line[0], line[1]);
	cv::Point right_top(line[2], line[3]);
	float slope = float(right_bottom.y - right_top.y) / float(right_bottom.x - right_top.x);
	float y_intercept = right_top.y - slope * right_top.x;
	float x = y_pt / slope - y_intercept;
	int x_pt = static_cast<int>(x);
	return cv::Point(x_pt, y_pt);
}


cv::Vec4i FeatureExtractor::find_lowest_point(std::vector<cv::Vec4i>& input, int l_threshold, int r_threshold) {
	cv::Mat lines(input);
	std::vector<cv::Point> points;
	for (int i = 0; i < lines.rows; i++) {
		//std::cout << lines.at<cv::Vec4i>(i) << std::endl;
		cv::Vec4i pts = lines.at<cv::Vec4i>(i);
		cv::Point x1y1(pts[0], pts[1]);
		cv::Point x2y2(pts[2], pts[3]);
		points.push_back(x1y1);
		points.push_back(x2y2);
	}

	cv::Mat points_matrix(points.size(), 2, CV_32S);;
	for (int i = 0; i < points.size(); i++) {
		points_matrix.at<int>(i, 0) = points[i].x;
		points_matrix.at<int>(i, 1) = points[i].y;
	}

	cv::Point l_min_pt(points_matrix.at<int>(0, 0), 0);
	cv::Point r_min_pt(points_matrix.at<int>(1, 0), 0);
	for (int i = 0; i < points_matrix.rows; i++) {
		int x_pt = points_matrix.at<int>(i, 0);
		int y_pt = points_matrix.at<int>(i, 1);

		if (x_pt < l_threshold && y_pt > l_min_pt.y) {
			l_min_pt = cv::Point(x_pt, y_pt);
		}
		else if (x_pt > r_threshold&& y_pt > r_min_pt.y) {
			r_min_pt = cv::Point(x_pt, y_pt);
		}
	}
	cv::Vec4i top_line;
	top_line[0] = l_min_pt.x;
	top_line[1] = l_min_pt.y;
	top_line[2] = r_min_pt.x;
	top_line[3] = r_min_pt.y;
	return top_line;
}

cv::Vec4i FeatureExtractor::find_highest_point(std::vector<cv::Vec4i>& input, int l_threshold, int r_threshold) {
	cv::Mat lines(input);
	std::vector<cv::Point> points;
	for (int i = 0; i < lines.rows; i++) {
		//std::cout << lines.at<cv::Vec4i>(i) << std::endl;
		cv::Vec4i pts = lines.at<cv::Vec4i>(i);
		cv::Point x1y1(pts[0], pts[1]);
		cv::Point x2y2(pts[2], pts[3]);
		points.push_back(x1y1);
		points.push_back(x2y2);
	}

	cv::Mat points_matrix(points.size(), 2, CV_32S);;
	for (int i = 0; i < points.size(); i++) {
		points_matrix.at<int>(i, 0) = points[i].x;
		points_matrix.at<int>(i, 1) = points[i].y;
	}

	cv::Point l_max_pt(points_matrix.at<int>(0, 0), 10000);
	cv::Point r_max_pt(points_matrix.at<int>(1, 0), 10000);
	for (int i = 0; i < points_matrix.rows; i++) {
		int x_pt = points_matrix.at<int>(i, 0);
		int y_pt = points_matrix.at<int>(i, 1);

		if (x_pt < l_threshold && y_pt < l_max_pt.y) {
			l_max_pt = cv::Point(x_pt, y_pt);
		}
		else if (x_pt > r_threshold && y_pt < r_max_pt.y) {
			r_max_pt = cv::Point(x_pt, y_pt);
		}
	}
	cv::Vec4i top_line;
	top_line[0] = l_max_pt.x;
	top_line[1] = l_max_pt.y;
	top_line[2] = r_max_pt.x;
	top_line[3] = r_max_pt.y;
	return top_line;
}


cv::Vec4i FeatureExtractor::find_lowest_point(std::vector<cv::Vec4i>& input, int middle_pt, bool use_middle) {
	cv::Mat lines(input);
	std::vector<cv::Point> points;
	for (int i = 0; i < lines.rows; i++) {
		//std::cout << lines.at<cv::Vec4i>(i) << std::endl;
		cv::Vec4i pts = lines.at<cv::Vec4i>(i);
		cv::Point x1y1(pts[0], pts[1]);
		cv::Point x2y2(pts[2], pts[3]);
		points.push_back(x1y1);
		points.push_back(x2y2);
	}

	cv::Mat points_matrix(points.size(), 2, CV_32S);;
	for (int i = 0; i < points.size(); i++) {
		points_matrix.at<int>(i, 0) = points[i].x;
		points_matrix.at<int>(i, 1) = points[i].y;
	}

	cv::Point l_min_pt(points_matrix.at<int>(0, 0), 0);
	cv::Point r_min_pt(points_matrix.at<int>(1, 0), 0);
	for (int i = 0; i < points_matrix.rows; i++) {
		int x_pt = points_matrix.at<int>(i, 0);
		int y_pt = points_matrix.at<int>(i, 1);

		if (x_pt < middle_pt && y_pt > l_min_pt.y) {
			l_min_pt = cv::Point(x_pt, y_pt);
		}
		else if (x_pt > middle_pt&& y_pt > r_min_pt.y) {
			r_min_pt = cv::Point(x_pt, y_pt);
		}
	}
	cv::Vec4i top_line;
	top_line[0] = l_min_pt.x;
	top_line[1] = l_min_pt.y;
	top_line[2] = r_min_pt.x;
	top_line[3] = r_min_pt.y;
	return top_line;
}

cv::Vec4i FeatureExtractor::find_highest_point(std::vector<cv::Vec4i>& input, int middle_pt, bool use_middle) {
	cv::Mat lines(input);
	std::vector<cv::Point> points;
	for (int i = 0; i < lines.rows; i++) {
		//std::cout << lines.at<cv::Vec4i>(i) << std::endl;
		cv::Vec4i pts = lines.at<cv::Vec4i>(i);
		cv::Point x1y1(pts[0], pts[1]);
		cv::Point x2y2(pts[2], pts[3]);
		points.push_back(x1y1);
		points.push_back(x2y2);
	}

	cv::Mat points_matrix(points.size(), 2, CV_32S);;
	for (int i = 0; i < points.size(); i++) {
		points_matrix.at<int>(i, 0) = points[i].x;
		points_matrix.at<int>(i, 1) = points[i].y;
	}

	cv::Point l_max_pt(points_matrix.at<int>(0,0), 10000);
	cv::Point r_max_pt(points_matrix.at<int>(1, 0), 10000);
	for (int i = 0; i < points_matrix.rows; i++) {
		int x_pt = points_matrix.at<int>(i, 0);
		int y_pt = points_matrix.at<int>(i, 1);
		
		if (x_pt < middle_pt && y_pt < l_max_pt.y) {
			l_max_pt = cv::Point(x_pt, y_pt);
		}
		else if (x_pt > middle_pt && y_pt < r_max_pt.y) {
			r_max_pt = cv::Point(x_pt, y_pt);
		}
	}
	cv::Vec4i top_line;
	top_line[0] = l_max_pt.x;
	top_line[1] = l_max_pt.y;
	top_line[2] = r_max_pt.x;
	top_line[3] = r_max_pt.y;
	return top_line;
}

cv::Mat FeatureExtractor::lane_detect(cv::Mat& input_frame, int l_threshold, int r_threshold, std::vector<std::pair<float, float>>& roi,ConfigurationParameters& config, bool remove_between_lanes) {
	// Convert to HLS color space
	cv::Mat hls_im;
	cv::cvtColor(input_frame, hls_im, cv::COLOR_BGR2HLS);

	// Get RGB img
	cv::Mat rgb_im = input_frame.clone();

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
	cv::GaussianBlur(combined, blurred, { 3,3 }, 0);

	// Edge detect
	cv::Mat edges;
	cv::Canny(blurred, edges, 100, 190);

	// From the edges, remove unnecessary edges
	// propose a region of interest
	cv::Mat roi_im = this->propose_roi(
		edges,
		roi,
		false
	);

	if (remove_between_lanes) {
		cv::Mat inner_mask = this->create_inner_cover_mask(roi_im, config.inner_roi);
		roi_im.convertTo(roi_im, CV_8UC1);
		inner_mask.convertTo(inner_mask, CV_8UC1);
		roi_im = this->remove_middle_polygons(roi_im, inner_mask);
	}

	//this->show_image(roi_im, 1, 1, 5000);

	rgb_im = this->get_lanes(roi_im, rgb_im);
	return rgb_im;
}

cv::Mat FeatureExtractor::lane_detect(cv::Mat& input_frame, std::vector<std::pair<float, float>>& roi) {
	// Convert to HLS color space
	cv::Mat hls_im;
	cv::cvtColor(input_frame, hls_im, cv::COLOR_BGR2HLS);

	// Get RGB img
	cv::Mat rgb_im = input_frame.clone();

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
	cv::GaussianBlur(combined, blurred, { 3,3 }, 0);

	// Edge detect
	cv::Mat edges;
	cv::Canny(blurred, edges, 100, 190);

	// From the edges, remove unnecessary edges
	// propose a region of interest
	cv::Mat roi_im = this->propose_roi(
		edges,
		roi,
		false
	);
	//this->show_image(roi_im, 1, 1, 5000);

	rgb_im = this->get_lanes(roi_im, rgb_im);
	return rgb_im;
}

cv::Mat FeatureExtractor::lane_detect(cv::Mat& input_frame) {
	// Convert to HLS color space
	cv::Mat hls_im;
	cv::cvtColor(input_frame, hls_im, cv::COLOR_BGR2HLS);

	// Get RGB img
	cv::Mat rgb_im = input_frame.clone();

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
	cv::GaussianBlur(combined, blurred, { 3,3 }, 0);

	// Edge detect
	cv::Mat edges;
	cv::Canny(blurred, edges, 100, 190);

	// From the edges, remove unnecessary edges
	// propose a region of interest
	cv::Mat roi_im = this->propose_roi(edges,
		0, 1,
		1, 1,
		0.518, 0.59,
		0.518, 0.59,
		false
	);
	//this->show_image(roi_im, 1, 1, 5000);

	rgb_im = this->get_lanes(roi_im, rgb_im);
	return rgb_im;
}

void FeatureExtractor::show_image(cv::Mat& img, int x_window, int y_window, int delay) {
	cv::namedWindow("output", cv::WINDOW_NORMAL);
	cv::resizeWindow("output", img.cols / x_window, img.rows / y_window);
	cv::imshow("output", img);
	cv::waitKey(delay);
}

std::pair<std::vector<cv::Mat>, std::vector<int>> FeatureExtractor::load_dataset(std::string car_ds_loc, std::string non_car_ds,bool debug,int num_imgs) {
	
	std::vector<cv::Mat> vehicles_arr = this->load_images(car_ds_loc,debug,num_imgs);
	std::vector<int> vehicle_labels(vehicles_arr.size(), 1);

	std::vector<cv::Mat> non_vehicles_arr = this->load_images(non_car_ds,debug,num_imgs);
	std::vector<int> non_vehicle_labels(non_vehicles_arr.size(), -1);

	// Concat matrices
	std::vector<cv::Mat> x_data;
	x_data.reserve(vehicles_arr.size() + non_vehicles_arr.size());
	x_data.insert(x_data.end(), vehicles_arr.begin(), vehicles_arr.end());
	x_data.insert(x_data.end(), non_vehicles_arr.begin(), non_vehicles_arr.end());
	std::vector<int> y_data;
	std::cout << "Vehicle label size = " << vehicle_labels.size() << std::endl;
	std::cout << "None Vehicle label size = " << non_vehicle_labels.size() << std::endl;
	y_data.reserve(vehicle_labels.size() + non_vehicle_labels.size());
	y_data.insert(y_data.end(), vehicle_labels.begin(), vehicle_labels.end());
	y_data.insert(y_data.end(), non_vehicle_labels.begin(), non_vehicle_labels.end());
	std::pair<std::vector<cv::Mat>, std::vector<int>> dataset = { x_data,y_data };
	return dataset;
}

std::pair<std::vector<cv::Mat>, std::vector<int>> FeatureExtractor::load_dataset(std::string dataset_loc,int label, bool debug,int num_imgs) {
	std::vector<cv::Mat> x_data = this->load_images(dataset_loc, debug,num_imgs);
	std::vector<int> y_data(x_data.size(), label);
	std::pair<std::vector<cv::Mat>, std::vector<int>> dataset = { x_data,y_data };
	return dataset;
}


std::vector<cv::Mat> FeatureExtractor::load_images(std::string dataset_loc,bool debug,int num_imgs) {
	int count = 0;
	std::vector<cv::Mat> images;
	for (const auto& entry : std::filesystem::directory_iterator(dataset_loc)) {
		// Go into various dirs
		std::string curr_item_path = entry.path().string();
		std::vector<std::string> dir_items = this->split(curr_item_path, '/');
		std::string curr_item = dir_items[dir_items.size() - 1];
		if (curr_item != ".DS_Store") {
			// go into each of those directories
			for (const auto& im_file : std::filesystem::directory_iterator(curr_item_path)) {
				std::string curr_im_path = im_file.path().string();
				std::vector<std::string> im_dir_items = this->split(curr_im_path, '\\');
				std::string curr_im = im_dir_items[im_dir_items.size() - 1];
				if (curr_im != ".DS_Store") {
					if (count == num_imgs)
						return images;
					// Open images
					cv::Mat curr_im_mat = cv::imread(curr_im_path);
					images.push_back(curr_im_mat);
				}
				if(debug)
					count++;
			}
		}
	}
	return images;
}

std::vector<cv::Mat> FeatureExtractor::featurize_dataset(std::vector<cv::Mat>& dataset,bool debug) {
	// HOG Params
	cv::Size window_stride = { 8,8};
	cv::Size padding = { 0,0 };
	std::vector<float> descriptors;
	std::vector<cv::Point> location_pts;
	cv::HOGDescriptor hog;
	hog.winSize = cv::Size(64,64);
	
	std::vector<cv::Mat> hog_ims;
	for (int i = 0; i < dataset.size(); i++) {
		
		cv::Mat curr_im = dataset[i];
		cv::Mat gray_im;
		cv::cvtColor(curr_im, gray_im, cv::COLOR_BGR2GRAY);

		// Smoothing the img
		//cv::GaussianBlur(gray_im, gray_im, cv::Size(7, 7), 10, 30);

		//location_pts.push_back(cv::Point(gray_im.cols / 2, gray_im.rows / 2));
		
		hog.compute(gray_im, descriptors,window_stride,padding,location_pts);
		int des_sz = descriptors.size();
		//cv::Mat test = this->get_hogdescriptor_visu(curr_im.clone(), descriptors, cv::Size(64, 64));
		//this->show_image(test, 1, 1, 5000);

		cv::Mat out = cv::Mat(descriptors).clone();
		hog_ims.push_back(out);
		if (debug) {
			printf("r of descriptors = %i -- c of descriptiors = %i\n", out.rows, out.cols);
			for (int item = 0; item < out.rows; item++) {
				printf("%f , ", out.at<float>(item));
			}
		}
	}
	return hog_ims;
}

// https://stackoverflow.com/questions/9435385/split-a-string-using-c11/9437426
std::vector<std::string> FeatureExtractor::split(const std::string& s, char delim) {
	std::stringstream ss(s);
	std::string item;
	std::vector<std::string> elems;
	while (std::getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}

std::string FeatureExtractor::get_name_num(std::string& file_loc) {
	std::vector<std::string> f_path_items = this->split(file_loc, '/');
	std::string f_name = f_path_items[f_path_items.size() - 1];
	std::vector<std::string> f_name_items = this->split(f_name, '.');
	std::string f_name_num = f_name_items[0];
	return f_name_num;
}

cv::Mat FeatureExtractor::normalize_dataset(cv::Mat& x_data) {
	cv::Mat norm_x_data;
	cv::normalize(x_data, norm_x_data, 1.0, 0.0, cv::NORM_INF);
	return norm_x_data;
}

void FeatureExtractor::train_svm(cv::Mat& x_data, cv::Mat& y_data,std::string model_fname) {
	printf("------{x_data size = %i}==={y_data size = %i}---------", x_data.rows, y_data.rows);
	printf("Training SVM\n");
	cv::Ptr<cv::ml::SVM> svm_model = cv::ml::SVM::create();
	// hyper param setup
	svm_model->setCoef0(0.0);
	svm_model->setDegree(3);
	svm_model->setGamma(0);
	svm_model->setNu(0.5);
	svm_model->setP(0.1);
	svm_model->setC(0.01);
	svm_model->setType(cv::ml::SVM::EPS_SVR);
	//svm_model->setType(cv::ml::SVM::C_SVC);
	//svm_model->setType(cv::ml::SVM::ONE_CLASS);
	svm_model->setKernel(cv::ml::SVM::LINEAR);
	svm_model->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-3));
	svm_model->train(x_data, cv::ml::ROW_SAMPLE, y_data);
	svm_model->save(model_fname);
	printf("-- Training Complete -- \n");
}

void FeatureExtractor::train_test_svm(const cv::Mat& x_train, const cv::Mat& y_train, 
									  const cv::Mat& x_test, const cv::Mat& y_test,
									  bool save_model, std::string model_fname) {
	printf(" -------- Training SVM ---------\n");
	printf("x_train size = %i\n", x_train.rows);
	printf("y_train size = %i\n", y_train.rows);
	printf("x_test size = %i\n", x_test.rows);
	printf("y_test size = %i\n", y_test.rows);
	cv::Ptr<cv::ml::SVM> svm_model = cv::ml::SVM::create();
	// hyper param setup
	svm_model->setCoef0(0.0);
	svm_model->setDegree(3);
	svm_model->setGamma(0);
	svm_model->setNu(0.5);
	svm_model->setP(0.1);
	svm_model->setC(0.01);
	svm_model->setType(cv::ml::SVM::EPS_SVR);
	//svm_model->setType(cv::ml::SVM::C_SVC);
	//svm_model->setType(cv::ml::SVM::ONE_CLASS);
	svm_model->setKernel(cv::ml::SVM::LINEAR);
	svm_model->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-3));
	svm_model->train(x_train, cv::ml::ROW_SAMPLE, y_train);
	if (save_model) {
		svm_model->save(model_fname);
	}
	printf("-- Training Complete -- \n");

	cv::Mat predictions;
	svm_model->predict(x_test, predictions);

	//std::cout << predictions << std::endl;
	//std::cout << "\n";
	//std::cout << y_test << std::endl;

	// Score the test
	float correct = 0;
	for (int i = 0; i < predictions.rows; i++) {
		float pred = predictions.at<float>(i);
		//std::cout << "Pred = " << pred << std::endl;
		int actual = y_test.at<int>(i);
		//std::cout << "Actual = " << actual << std::endl;
		if (pred == actual) {
			correct++;
		}
	}
	float score = correct / static_cast<float>(predictions.rows);
	//std::cout << "SVM Test Accuracy = " << score << std::endl;
	printf("SVM Test Accuracy = %f\n", score);
}

cv::Ptr<cv::ml::TrainData> FeatureExtractor::train_test_split(cv::Mat& x_data, cv::Mat& y_data, int test_size) {
	int train_size = x_data.rows - test_size;
	cv::Ptr<cv::ml::TrainData> dataset = cv::ml::TrainData::create(x_data, cv::ml::ROW_SAMPLE, y_data);
	dataset->setTrainTestSplit(train_size);
	return dataset;
}

std::pair<cv::Mat, cv::Mat> FeatureExtractor::prepare_training_data(std::vector<cv::Mat>& x_data, std::vector<int>& y_data) {
	// Convert x_data to mat
	int rs = x_data.size();
	int cs = std::max(x_data[0].rows, x_data[0].cols);
	cv::Mat x_data_mat(rs,cs,CV_32FC1);
	printf("x_data shape = { r = %i , c = %i}\n", x_data_mat.rows, x_data_mat.cols);
	cv::Mat current_sample(1, std::max(x_data[0].rows, x_data[0].cols), CV_32FC1);
	for (int i = 0; i < x_data.size(); i++) {
		//x_data[i].copyTo(x_data_mat.row(i));
		CV_Assert(x_data[i].cols == 1 ||
			x_data[i].rows == 1);
		if (x_data[i].cols == 1)
		{
			transpose(x_data[i], current_sample);
			current_sample.copyTo(x_data_mat.row(i));
		}
		else if (x_data[i].rows == 1)
		{
			x_data[i].copyTo(x_data_mat.row(i));
		}
	}
	// convert y_data
	std::pair<cv::Mat, cv::Mat> ret = {x_data_mat,cv::Mat(y_data)};
	return ret;
}

// https://docs.opencv.org/3.4/d0/df8/samples_2cpp_2train_HOG_8cpp-example.html#a70
std::vector<float> FeatureExtractor::get_svm_detector(std::string model_loc) {
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
	float sz = support_vectors.cols + 1;
	std::vector<float> hog_detector;
	hog_detector.resize(sz);
	memcpy(&hog_detector[0], support_vectors.ptr(), support_vectors.cols * sizeof(hog_detector[0]));
	hog_detector[support_vectors.cols] = (float)-rho;
	return hog_detector;
}

std::pair<cv::Ptr<cv::ml::SVM>, std::vector<float>> FeatureExtractor::get_svm_detector(std::string model_loc,int class_num) {
	cv::Ptr<cv::ml::SVM> svm_model = cv::ml::SVM::load(model_loc);
	int sv_dim = svm_model->getVarCount();
	//cv::Ptr<cv::ml::SVM> svm_model = cv::ml::StatModel::load<cv::ml::SVM>(model_loc);
	cv::Mat support_vectors = svm_model->getSupportVectors();
	const int sv_total = support_vectors.rows;
	std::cout << "SV TOTAL = " << sv_total << std::endl;
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


std::vector<cv::Point> FeatureExtractor::detection_roi(cv::Mat& input, double top_l1, double top_l2,
																	   double top_r1, double top_r2,
																	   double bottom_l1, double bottom_l2,
																	   double bottom_r1, double bottom_r2, bool debug) {
	//input.convertTo(input, CV_32FC1);
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

	std::vector<cv::Point> roi = { top_left_pt ,top_right_pt,bottom_left_pt,bottom_right_pt };

	if (debug) {
		cv::Mat debug_im = input.clone();
		cv::line(debug_im, top_left_pt, bottom_left_pt, cv::Scalar(0, 0, 255), 2);
		cv::line(debug_im, top_right_pt, bottom_right_pt, cv::Scalar(0, 0, 255), 2);
		cv::line(debug_im, top_left_pt, top_right_pt, cv::Scalar(0, 0, 255), 2);
		cv::line(debug_im, bottom_left_pt, bottom_right_pt, cv::Scalar(0, 0, 255), 2);
		this->show_image(debug_im, 1, 1, 5000);
		cv::imwrite("roi_im.png", debug_im);
	}
	return roi;
}

cv::Mat FeatureExtractor::vehicle_detect(cv::Mat& img, cv::HOGDescriptor& detector, std::vector<cv::Point>& roi,bool include_all_bboxes) {
	cv::Rect roi_im(roi[0].x, roi[0].y, roi[1].x - roi[0].x, roi[2].y - roi[0].y);
	cv::Mat cropped_im = img(roi_im);
	std::vector<cv::Rect> found_locations;
	std::vector<double> confidence;
	detector.detectMultiScale(cropped_im, found_locations, confidence, 0.0, cv::Size(8, 8), cv::Size(0, 0), 1.2632, 2.0);//cv::Size(10,10), cv::Size(0, 0), 1.2632, 2.0);
	std::vector<float> confidence2;
	for (const double conf : confidence) {
		confidence2.push_back(static_cast<float>(conf));
	}
	std::vector<int> keep_vec;
	cv::dnn::dnn4_v20190902::MatShape kept_boxes;
	cv::dnn::NMSBoxes(found_locations, confidence2, 0.1f, 0.1f, keep_vec);//0.4f, 0.3f, keep_vec);

	if (include_all_bboxes) {
		for (cv::Rect r : found_locations) {
			cv::rectangle(cropped_im, r, cv::Scalar(0, 255, 0), 2);
		}
	}

	for (const int idx : keep_vec) {
		cv::Rect curr_rect = found_locations[idx];
		cv::rectangle(cropped_im, curr_rect, cv::Scalar(0, 0, 255), 2);
	}
	
	cropped_im.copyTo(img(roi_im));
	return img;
}

std::vector<cv::Rect> FeatureExtractor::vehicle_detect_bboxes(cv::Mat& img, cv::HOGDescriptor& detector, std::vector<cv::Point>& roi, bool include_all_bboxes) {
	cv::Rect roi_im(roi[0].x, roi[0].y, roi[1].x - roi[0].x, roi[2].y - roi[0].y);
	cv::Mat cropped_im = img(roi_im);
	std::vector<cv::Rect> found_locations;
	std::vector<double> confidence;
	detector.detectMultiScale(cropped_im, found_locations, confidence, 0.0, cv::Size(8, 8), cv::Size(0, 0), 1.2632, 2.0);//cv::Size(10,10), cv::Size(0, 0), 1.2632, 2.0);
	std::vector<float> confidence2;
	for (const double conf : confidence) {
		confidence2.push_back(static_cast<float>(conf));
	}
	std::vector<int> keep_vec;
	cv::dnn::dnn4_v20190902::MatShape kept_boxes;
	cv::dnn::NMSBoxes(found_locations, confidence2, 0.1f, 0.1f, keep_vec);//0.4f, 0.3f, keep_vec);

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

std::vector<cv::Rect> FeatureExtractor::respace(std::vector<cv::Rect>& bboxes,cv::Rect& roi) {
	std::vector<cv::Rect> adjusted_bboxes;
	for (cv::Rect curr_rect : bboxes) {
		curr_rect.x += roi.x;
		curr_rect.y += roi.y;
		adjusted_bboxes.push_back(curr_rect);
	}
	return adjusted_bboxes;
}

void FeatureExtractor::display_num_vehicles(cv::Mat& img,std::vector<cv::Rect>& bboxes) {
	int num_bboxes = bboxes.size();
	std::string s_num_boxes = std::to_string(num_bboxes);
	cv::putText(
		img, 
		cv::String("Number of Vehicles in view: " + s_num_boxes), 
		cv::Point(10, img.rows-30),
		cv::FONT_HERSHEY_DUPLEX,
		0.5, 
		cv::Scalar(0, 0, 255)
	);
}

std::vector<cv::Rect> FeatureExtractor::sliding_window(cv::Mat& img, cv::Size& win_stride, cv::Size& window_size, float scale, const cv::Ptr <cv::ml::SVM>& model) {
	cv::Mat gray_im;
	cv::cvtColor(img, gray_im, cv::COLOR_BGR2GRAY);
	std::vector<cv::Rect> bboxes;
	
	cv::Mat temp = gray_im.clone();
	cv::Mat dest = gray_im.clone();
	cv::HOGDescriptor hog;
	hog.winSize = cv::Size(64, 64);
	
	bool downsample = true;

	while (downsample) {
		std::vector<float> descriptors;
		std::vector<cv::Point> loc_pts;

		for (int row = 0; row+ win_stride.height <= gray_im.rows - win_stride.height; row+=win_stride.height) {
			for (int col = 0; col+win_stride.width <= gray_im.cols - win_stride.width; col+=win_stride.width) {
				cv::Rect curr_window(col, row, window_size.width, window_size.height);
				if (curr_window.x >= 0 && curr_window.y >= 0 && curr_window.width + curr_window.x < img.cols && curr_window.height + curr_window.y < img.rows){
					cv::Mat curr_patch = dest(curr_window);
					hog.compute(curr_patch, descriptors, win_stride, cv::Size(0, 0), loc_pts);
					cv::Mat prediction;
					cv::Mat desc(descriptors);
					//desc = desc.reshape(1, 1);
					cv::transpose(desc, desc);
					desc.convertTo(desc, CV_32F);
					model->predict(desc, prediction);
					if (prediction.at<int>(0) == 1) {
						//cv::Rect adjusted_rect(col,row, window_size.width, window_size.height)
						bboxes.push_back(curr_window);
					}
				}
			}
		}
		cv::resize(temp, dest, cv::Size(temp.cols / scale, temp.rows / scale));
		if (dest.rows <= window_size.height || dest.cols <= window_size.width) {
			break;
		}
	}
	return bboxes;
}