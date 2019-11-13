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

	std::vector<cv::Vec4i> hough_lines;
	for (int i = 0; i < lines.size(); i++) {
		cv::Vec4i pts = lines[i];
		cv::Point x1y1(pts[0], pts[1]);
		cv::Point x2y2(pts[2], pts[3]);
		hough_lines.push_back(pts);
		std::cout << "xy  = " << x1y1 << std::endl;
		std::cout << "xy  = " << x2y2 << std::endl;
		cv::line(output, x1y1,x2y2, cv::Scalar(0, 0, 255));
	}

	std::cout << "before finding highest point function \n";
	// Look for highest point
	cv::Vec4i top_line = this->find_highest_point(hough_lines);
	cv::line(output, cv::Point(top_line[0], top_line[1]), cv::Point(top_line[2], top_line[3]), cv::Scalar(0, 0, 255));

	return output;
}

cv::Vec4i FeatureExtractor::find_highest_point(std::vector<cv::Vec4i>& input) {
	cv::Mat lines(input);
	std::vector<cv::Point> points;
	std::cout << "in finding highest point function \n";
	for (int i = 0; i < lines.rows; i++) {
		//std::cout << lines.at<cv::Vec4i>(i) << std::endl;
		cv::Vec4i pts = lines.at<cv::Vec4i>(i);
		cv::Point x1y1(pts[0], pts[1]);
		cv::Point x2y2(pts[2], pts[3]);
		points.push_back(x1y1);
		points.push_back(x2y2);
	}

	cv::Mat points_matrix(points.size(), 2, CV_32S);
	std::cout << "ROWS = " << points_matrix.rows << std::endl;
	std::cout << "COLS = " << points_matrix.cols << std::endl;
	for (int i = 0; i < points.size(); i++) {
		//std::cout << points[i] << std::endl;
		//std::cout << points[i].x << std::endl;
		//std::cout << points[i].y << std::endl;
		points_matrix.at<int>(i, 0) = points[i].x;
		points_matrix.at<int>(i, 1) = points[i].y;
	}

	//std::cout << points_matrix << std::endl;

	cv::Mat x_vals = points_matrix.col(0);
	std::cout << x_vals << std::endl;

	cv::Mat1i sorted_points_idx;
	cv::sortIdx(x_vals, sorted_points_idx, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
	cv::sort(x_vals, x_vals, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);

	cv::Mat sorted_points_matrix(points.size(), 2, CV_32S);
	std::cout << std::endl;
	for (int i = 0; i < points_matrix.rows; i++) {
		points_matrix.row(sorted_points_idx(i)).copyTo(sorted_points_matrix.row(i));
		
		//points_matrix.at<int>(i, 0) = sorted_points_matrix.at<int>(i);
	}

	std::cout << "Sorted POINTS \n";
	std::cout << sorted_points_matrix << std::endl;

	cv::Point l_max_pt(points_matrix.at<int>(0,0), points_matrix.at<int>(0, 1));
	cv::Point r_max_pt(points_matrix.at<int>(1, 0), points_matrix.at<int>(1, 1));
	std::cout << l_max_pt << " , " << r_max_pt << std::endl;
	for (int i = 0; i < points_matrix.rows; i++) {
		int x_pt = points_matrix.at<int>(i, 0);
		int y_pt = points_matrix.at<int>(i, 1);
		
		if (x_pt < 650 && y_pt < l_max_pt.y) {
			std::cout << "LEFT X = " << x_pt << " ---- Y = " << y_pt << std::endl;
			l_max_pt = cv::Point(x_pt, y_pt);
		}
		else if (x_pt > 650 && y_pt < r_max_pt.y) {
			std::cout << "RIGHT X = " << x_pt << " ---- Y = " << y_pt << std::endl;
			r_max_pt = cv::Point(x_pt, y_pt);
		}
	}
	std::cout << l_max_pt << " , " << r_max_pt << std::endl;

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