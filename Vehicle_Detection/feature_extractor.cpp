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
													  double bottom_l1, double bottom_l2,
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

	std::vector<cv::Vec4i> hough_lines;
	for (int i = 0; i < lines.size(); i++) {
		cv::Vec4i pts = lines[i];
		cv::Point x1y1(pts[0], pts[1]);
		cv::Point x2y2(pts[2], pts[3]);
		hough_lines.push_back(pts);
		//cv::line(output, x1y1,x2y2, cv::Scalar(0, 0, 255));
	}
	// Look for highest point
	cv::Vec4i top_line = this->find_highest_point(hough_lines);
	cv::line(output, cv::Point(top_line[0], top_line[1]), cv::Point(top_line[2], top_line[3]), cv::Scalar(0, 0, 255),2);

	// Look for the lowest point
	cv::Vec4i min_points = this->find_lowest_point(hough_lines);

	// Extrapolate the right lane
	cv::Vec4i current_right_lane = { min_points[2],min_points[3],top_line[2],top_line[3]};
	std::cout << " Running \n";
	cv::Point adjusted_right_min = this->extrapolate_line(current_right_lane, min_points[1]);
	std::cout << adjusted_right_min << std::endl;

	cv::line(
		output, 
		cv::Point(min_points[0], min_points[1]), 
		cv::Point(top_line[0], top_line[1]), 
		cv::Scalar(0, 0, 255),
		2
	);

	cv::line(
		output, 
		cv::Point(adjusted_right_min.x, adjusted_right_min.y), 
		cv::Point(top_line[2], top_line[3]), 
		cv::Scalar(0, 0, 255),
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
	cv::fillPoly(overlay, corner_list, &num_points, num_polygons, cv::Scalar(255));
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


cv::Vec4i FeatureExtractor::find_lowest_point(std::vector<cv::Vec4i>& input) {
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

		if (x_pt < 650 && y_pt > l_min_pt.y) {
			l_min_pt = cv::Point(x_pt, y_pt);
		}
		else if (x_pt > 650 && y_pt > r_min_pt.y) {
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

cv::Vec4i FeatureExtractor::find_highest_point(std::vector<cv::Vec4i>& input) {
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
		
		if (x_pt < 650 && y_pt < l_max_pt.y) {
			l_max_pt = cv::Point(x_pt, y_pt);
		}
		else if (x_pt > 650 && y_pt < r_max_pt.y) {
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

std::pair<std::vector<cv::Mat>, std::vector<int>> FeatureExtractor::load_dataset(std::string car_ds_loc, std::string non_car_ds) {
	
	std::vector<cv::Mat> vehicles_arr = this->load_images(car_ds_loc);
	//cv::Size v_labels_sz = { 1, (int)vehicles_arr.size() };
	//cv::Mat vehicle_labels = cv::Mat::ones(v_labels_sz, CV_32F);
	std::vector<int> vehicle_labels(vehicles_arr.size(), 1);

	std::vector<cv::Mat> non_vehicles_arr = this->load_images(non_car_ds);
	//cv::Size nv_labels_sz = { 1, (int)non_vehicles_arr.size() };
	//cv::Mat non_vehicle_labels = cv::Mat::zeros(nv_labels_sz, CV_32F);
	std::vector<int> non_vehicle_labels(non_vehicles_arr.size(), 0);
	

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

std::vector<cv::Mat> FeatureExtractor::load_images(std::string dataset_loc) {
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
					// Open images
					cv::Mat curr_im_mat = cv::imread(curr_im_path);
					images.push_back(curr_im_mat);
				}
			}
		}
	}
	return images;
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