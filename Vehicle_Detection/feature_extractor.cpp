/// FeatureExtractor
/// By Gohur, Umair, Will
/// Extracts features of input images. These features include both
/// cars and lane lines. Lane lines are detected using Hough Transform
/// and are highlighted (traced) with a red line and filled in between. 
/// Cars features are loaded. 
/// Pre: Input images of cars and lane lines
/// Post: Output of lane lines and cars with bound boxes.
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
	cv::inRange(img, lower_bound, upper_bound, mask);
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

/// <summary>
/// remove_middle_polygons
/// the goal of this method is to iterate through
/// the mask and find the pixels within the bounds of 
/// the mask and set them to zero
/// </summary>
/// <param name="edge_im"></param>
/// <param name="mask"></param>
/// <returns></returns>
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

/// <summary>
/// Creates inner cover mask
/// </summary>
/// <param name="input"></param>
/// <param name="roi"></param>
/// <param name="debug"></param>
/// <returns></returns>
cv::Mat FeatureExtractor::create_inner_cover_mask(cv::Mat& input, std::vector<std::pair<float, float>>& roi, bool debug) {

	// define ROI point locations via ROI of percentages of image dims
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
	// define polygon size
	int num_points = 4;
	int num_polygons = 1;

	cv::Mat mask = cv::Mat::zeros(cv::Size(input.cols, input.rows), CV_64F);

	// Debug Draw Methods
	if (debug) {
		cv::line(input, bottom_left_pt, top_right_pt, 255);
		cv::line(input, bottom_right_pt, top_left_pt, 255);
		this->show_image(input, 1, 1, 5000);
	}
	// fill mask of zeros based on ROI points with 255 or white
	cv::fillPoly(mask, corner_list, &num_points, num_polygons, cv::Scalar(255));
	return mask;
}

/// <summary>
/// Propose the ROI, the ROI allows for our program to compute faster by specifying a certain region
/// to focus our search on. 
/// Pre: Need an image
/// Post: Returns the region of interest in image
/// </summary>
/// <param name="input">input image</param>
/// <param name="roi">Region of interest</param>
/// <param name="debug"></param>
/// <returns>The region that we are going to slide through</returns>
cv::Mat FeatureExtractor::propose_roi(cv::Mat& input, std::vector<std::pair<float, float>>& roi, bool debug) {
	input.convertTo(input, CV_64F);

	// define ROI point locations via ROI of percentages of image dims
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
	// remove the rest of the image:
	// perform an AND operation to remove everything that isn't within the polygon
	cv::bitwise_and(input, mask, output);
	return output;
}

/// <summary>
/// Proposes the ROI using pre-defined region selection
/// Pre: Input image and specified locations
/// Post: Outputs the specified region of interest
/// </summary>
/// <param name="input">input image</param>
/// <param name="top_l1">coordinate</param>
/// <param name="top_l2">coordinate</param>
/// <param name="top_r1">coordinate</param>
/// <param name="top_r2">coordinate</param>
/// <param name="bottom_l1">coordinate</param>
/// <param name="bottom_l2">coordinate</param>
/// <param name="bottom_r1">coordinate</param>
/// <param name="bottom_r2">coordinate</param>
/// <param name="debug"></param>
/// <returns>returns the specifed region of interest</returns>
cv::Mat FeatureExtractor::propose_roi(cv::Mat& input, double top_l1, double top_l2,
	double top_r1, double top_r2,
	double bottom_l1, double bottom_l2,
	double bottom_r1, double bottom_r2, bool debug) {
	input.convertTo(input, CV_64F);

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

	cv::Mat mask = cv::Mat::zeros(cv::Size(input.cols, input.rows), CV_64F);

	// Debug Draw Methods
	if (debug) {
		cv::line(input, bottom_left_pt, top_right_pt, 255);
		cv::line(input, bottom_right_pt, top_left_pt, 255);
		this->show_image(input, 1, 1, 5000);
	}

	cv::fillPoly(mask, corner_list, &num_points, num_polygons, cv::Scalar(255));
	cv::Mat output(cv::Size(mask.cols, mask.rows), CV_64F);
	// remove the rest of the image:
	// perform an AND operation to remove everything that isn't within the polygon
	cv::bitwise_and(input, mask, output);
	return output;
}

/// <summary>
/// Performs a Hough Transform and obtains the lane lines
/// A connection line is calculated to see the extent of the
/// detection. Additionally, the lanes are extrapolated
/// Preconditions: Input image and specified locations
/// Postconditions: Outputs the specified region of interest
/// </summary>
cv::Mat FeatureExtractor::get_lanes(ConfigurationParameters& config, cv::Mat& input, cv::Mat& output, int l_threshold, int r_threshold) {
	input.convertTo(input, CV_8UC1);
	std::vector<cv::Vec4i> lines;
	cv::HoughLinesP(
		input,
		lines,
		config.rho,
		config.theta,
		config.threshold,
		config.minLineLength,
		config.maxLineGap
	);
	if (lines.size() == 0) {
		return output;
	}

	std::vector<cv::Vec4i> hough_lines;
	for (int i = 0; i < lines.size(); i++) {
		cv::Vec4i pts = lines[i];
		cv::Point x1y1(pts[0], pts[1]);
		cv::Point x2y2(pts[2], pts[3]);
		hough_lines.push_back(pts);
	}

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
	cv::Point adjusted_right_min = this->extrapolate_line(current_right_lane, min_points[1]);

	// Drawing the lane lines
	this->draw_lane_lines(output, min_points, adjusted_right_min, top_line);

	// create the transparent overlay between lanes
	this->draw_lane_overlay(output, min_points, adjusted_right_min, top_line);
	return output;
}

/// <summary>
/// Get's lane lines using Hough Transform
/// Pre: Accepts image input and thresholds
/// Post: Image with lane lines that were detected with hough transform 
/// </summary>
/// <param name="input">Input image</param>
/// <param name="output">output image</param>
/// <param name="l_threshold"></param>
/// <param name="r_threshold"></param>
/// <returns></returns>
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

	// Collect hough lines from probabalistic hough transform
	std::vector<cv::Vec4i> hough_lines;
	for (int i = 0; i < lines.size(); i++) {
		cv::Vec4i pts = lines[i];
		cv::Point x1y1(pts[0], pts[1]);
		cv::Point x2y2(pts[2], pts[3]);
		hough_lines.push_back(pts);
	}

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
	cv::Point adjusted_right_min = this->extrapolate_line(current_right_lane, min_points[1]);

	// Drawing the lane lines
	this->draw_lane_lines(output, min_points, adjusted_right_min, top_line);

	// create the transparent overlay between lanes
	this->draw_lane_overlay(output, min_points, adjusted_right_min, top_line);
	return output;
}

/// <summary>
/// gets lanes with predefined threshold params
/// </summary>
/// Preconditions:		input and output image should be given
/// Postconditions:		img with detected lane lines displayed
/// <param name="input"></param>
/// <param name="output"></param>
/// <returns></returns>
cv::Mat FeatureExtractor::get_lanes(cv::Mat& input, cv::Mat& output) {
	input.convertTo(input, CV_8UC1);
	std::vector<cv::Vec4i> lines;
	int rho = 1;
	double theta = M_PI / 180;
	int threshold = 20;
	int minLineLength = 20;
	int maxLineGap = 300;
	cv::HoughLinesP(input, lines, rho, theta, threshold, minLineLength, maxLineGap);
	// if no lanes were found then just return the output img
	if (lines.size() == 0) {
		return output;
	}

	std::vector<cv::Vec4i> hough_lines;
	for (int i = 0; i < lines.size(); i++) {
		cv::Vec4i pts = lines[i];
		cv::Point x1y1(pts[0], pts[1]);
		cv::Point x2y2(pts[2], pts[3]);
		hough_lines.push_back(pts);
	}

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
	cv::Point adjusted_right_min = this->extrapolate_line(current_right_lane, min_points[1]);

	// Drawing the lane lines
	this->draw_lane_lines(output, min_points, adjusted_right_min, top_line);

	// create the transparent overlay between lanes
	this->draw_lane_overlay(output, min_points, adjusted_right_min, top_line);
	return output;
}

/// <summary>
/// Draws the right and left lanes based on the found lines
/// </summary>
/// Preconditions:	The highest and lowest points must have been found and the
///					shorter lane should be extrapolated to take sure lane lines
///					are equivalent
/// Postconditions:	Lanes are draws on the output image
/// <param name="output"></param>
/// <param name="min_points"></param>
/// <param name="adjusted_right_min"></param>
/// <param name="top_line"></param>
/// <returns></returns>
void FeatureExtractor::draw_lane_lines(cv::Mat& output, cv::Vec4i& min_points, cv::Point& adjusted_right_min, cv::Vec4i& top_line) {
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
}

/// <summary>
/// Draws an overlay between the lane lines
/// </summary>
/// Preconditions:	The highest and lowest points must have been found and the
///					shorter lane should be extrapolated to take sure lane lines
///					are equivalent
/// Postconditions:	An partially transparent overlay is drawn on the img
/// <param name="output"></param>
/// <param name="min_points"></param>
/// <param name="adjusted_right_min"></param>
/// <param name="top_line"></param>
/// <returns></returns>
void FeatureExtractor::draw_lane_overlay(cv::Mat& output, cv::Vec4i& min_points, cv::Point& adjusted_right_min, cv::Vec4i& top_line) {
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
}

/// <summary>
/// extrapolate_line
/// extends a linear function by a given y point
/// Preconditions: A cv::Vec4 integer line and a y_pt
/// Postconditions: Return the new point with the updated x_pt
/// </summary>
/// <param name="line"></param>
/// <param name="y_pt"></param>
/// <returns></returns>
cv::Point FeatureExtractor::extrapolate_line(cv::Vec4i& line, int y_pt) {
	cv::Point right_bottom(line[0], line[1]);
	cv::Point right_top(line[2], line[3]);
	float slope = float(right_bottom.y - right_top.y) / float(right_bottom.x - right_top.x);
	float y_intercept = right_top.y - slope * right_top.x;
	float x = y_pt / slope - y_intercept;
	int x_pt = static_cast<int>(x);
	return cv::Point(x_pt, y_pt);
}


/// <summary>
/// find_lowest_point
/// Preconditions:	Vector of 4 value vectors representing hough lines must be
///					provided. This is pre-calculated in the get_lanes method.
///					Addtionally, a middle_pt threshold must be provided to represent
///					the location of the middle of the lane
///	Postconditions:	Returns a 4 point vector containing the location of the lowest point
///					of each of the lanes
/// </summary>
/// <param name="input"></param>
/// <param name="middle_pt"></param>
/// <param name="use_middle"></param>
/// <returns></returns>
cv::Vec4i FeatureExtractor::find_lowest_point(std::vector<cv::Vec4i>& input, int l_threshold, int r_threshold) {
	cv::Mat lines(input);
	std::vector<cv::Point> points;
	for (int i = 0; i < lines.rows; i++) {
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

/// <summary>
/// find_highest_point
/// Preconditions:	Vector of 4 value vectors representing hough lines must be
///					provided. This is pre-calculated in the get_lanes method.
///					Addtionally, a middle_pt threshold must be provided to represent
///					the location of the middle of the lane
///	Postconditions:	Returns a 4 point vector containing the location of the highest point
///					of each of the lanes
/// </summary>
/// <param name="input"></param>
/// <param name="middle_pt"></param>
/// <param name="use_middle"></param>
/// <returns></returns>
cv::Vec4i FeatureExtractor::find_highest_point(std::vector<cv::Vec4i>& input, int l_threshold, int r_threshold) {
	cv::Mat lines(input);
	std::vector<cv::Point> points;
	for (int i = 0; i < lines.rows; i++) {
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
		else if (x_pt > r_threshold&& y_pt < r_max_pt.y) {
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

/// <summary>
/// find_lowest_point
/// Preconditions:	Vector of 4 value vectors representing hough lines must be
///					provided. This is pre-calculated in the get_lanes method.
///					Addtionally, a middle_pt threshold must be provided to represent
///					the location of the middle of the lane
///	Postconditions:	Returns a 4 point vector containing the location of the lowest point
///					of each of the lanes
/// </summary>
/// <param name="input"></param>
/// <param name="middle_pt"></param>
/// <param name="use_middle"></param>
/// <returns></returns>
cv::Vec4i FeatureExtractor::find_lowest_point(std::vector<cv::Vec4i>& input, int middle_pt, bool use_middle) {
	cv::Mat lines(input);
	std::vector<cv::Point> points;
	for (int i = 0; i < lines.rows; i++) {
		// Get the vector of 4 points that define a line
		cv::Vec4i pts = lines.at<cv::Vec4i>(i);
		// create point obj for both ends of the line
		cv::Point x1y1(pts[0], pts[1]);
		cv::Point x2y2(pts[2], pts[3]);
		points.push_back(x1y1);
		points.push_back(x2y2);
	}

	// Set up 2 col matrix of points
	cv::Mat points_matrix(points.size(), 2, CV_32S);;
	for (int i = 0; i < points.size(); i++) {
		points_matrix.at<int>(i, 0) = points[i].x;
		points_matrix.at<int>(i, 1) = points[i].y;
	}
	// Set up initial lowest points for each lane
	cv::Point l_min_pt(points_matrix.at<int>(0, 0), 0);
	cv::Point r_min_pt(points_matrix.at<int>(1, 0), 0);
	// loop thru the matrix of points until we find the highest
	// y_pt on each side of the lane, recall that the higher the val, the lower in the img
	// the middle is set by middle_pt which defines the middle of the lane
	for (int i = 0; i < points_matrix.rows; i++) {
		int x_pt = points_matrix.at<int>(i, 0);
		int y_pt = points_matrix.at<int>(i, 1);
		if (x_pt < middle_pt && y_pt > l_min_pt.y) {
			l_min_pt = cv::Point(x_pt, y_pt);
		}
		else if (x_pt > middle_pt && y_pt > r_min_pt.y) {
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

/// <summary>
/// find_highest_point
/// Preconditions:	Vector of 4 value vectors representing hough lines must be
///					provided. This is pre-calculated in the get_lanes method.
///					Addtionally, a middle_pt threshold must be provided to represent
///					the location of the middle of the lane
///	Postconditions:	Returns a 4 point vector containing the location of the highest point
///					of each of the lanes
/// </summary>
/// <param name="input"></param>
/// <param name="middle_pt"></param>
/// <param name="use_middle"></param>
/// <returns></returns>
cv::Vec4i FeatureExtractor::find_highest_point(std::vector<cv::Vec4i>& input, int middle_pt, bool use_middle) {
	cv::Mat lines(input);
	std::vector<cv::Point> points;
	for (int i = 0; i < lines.rows; i++) {
		// Get the vector of 4 points that define a line
		cv::Vec4i pts = lines.at<cv::Vec4i>(i);
		// create point obj for both ends of the line
		cv::Point x1y1(pts[0], pts[1]);
		cv::Point x2y2(pts[2], pts[3]);
		points.push_back(x1y1);
		points.push_back(x2y2);
	}

	// Set up 2 col matrix of points
	cv::Mat points_matrix(points.size(), 2, CV_32S);;
	for (int i = 0; i < points.size(); i++) {
		points_matrix.at<int>(i, 0) = points[i].x;
		points_matrix.at<int>(i, 1) = points[i].y;
	}
	
	// Set up initial highest points for each lane
	cv::Point l_max_pt(points_matrix.at<int>(0, 0), 10000);
	cv::Point r_max_pt(points_matrix.at<int>(1, 0), 10000);
	// loop thru the matrix of points until we find the lowest
	// y_pt on each side of the lane, recall that the lower the val, the higher in the img
	// the middle is set by middle_pt which defines the middle of the lane
	for (int i = 0; i < points_matrix.rows; i++) {
		int x_pt = points_matrix.at<int>(i, 0);
		int y_pt = points_matrix.at<int>(i, 1);
		if (x_pt < middle_pt && y_pt < l_max_pt.y) {
			l_max_pt = cv::Point(x_pt, y_pt);
		}
		else if (x_pt > middle_pt&& y_pt < r_max_pt.y) {
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

/// <summary>
/// Lane Detect
/// Wrapper funtion that performs various methods including
/// white and yellow color detection, masking, ROI, and hough transform
/// Config based lane detection
/// lanes are detected based on the parameters
/// specified in the config.h config file.
/// </summary>
/// Preconditions:		configuration object should be passed in along with the
///						current image frame
/// Postconditions:		RGB image with detected lanes
cv::Mat FeatureExtractor::lane_detect(ConfigurationParameters& config, cv::Mat& input_frame) {
	// Get RGB img
	cv::Mat rgb_im = input_frame.clone();

	// Process the image
	// RGB -> HLS -> White/Yellow Mask -> Edge detect
	cv::Mat edges = this->extract_lane_colors(config, input_frame);

	// From the edges, remove unnecessary edges
	// propose a region of interest
	cv::Mat roi_im = this->propose_roi(
		edges,
		config.ld_roi,
		false
	);

	if (config.remove_between_lanes) {
		// Remove pixels that are within the lane lines
		cv::Mat inner_mask = this->create_inner_cover_mask(roi_im, config.inner_roi);
		roi_im.convertTo(roi_im, CV_8UC1);
		inner_mask.convertTo(inner_mask, CV_8UC1);
		roi_im = this->remove_middle_polygons(roi_im, inner_mask);
	}

	rgb_im = this->get_lanes(config, roi_im, rgb_im, config.l_threshold, config.r_threshold);
	return rgb_im;
}

/// <summary>
/// Wrapper funtion that performs various methods including
/// white and yellow color detection, masking, ROI, and hough transform
/// </summary>
/// Preconditions:		Current image frame and ROI of lanes should be determined and passed in
/// Postconditions:		RGB image with detected lanes
/// <param name="input_frame"></param>
/// <param name="roi"></param>
/// <returns></returns>
cv::Mat FeatureExtractor::lane_detect(cv::Mat& input_frame, std::vector<std::pair<float, float>>& roi) {
	// Get RGB img
	cv::Mat rgb_im = input_frame.clone();
	cv::Mat edges = this->extract_lane_colors(input_frame);

	// From the edges, remove unnecessary edges
	// propose a region of interest
	cv::Mat roi_im = this->propose_roi(
		edges,
		roi,
		false
	);

	rgb_im = this->get_lanes(roi_im, rgb_im);
	return rgb_im;
}

/// <summary>
/// Wrapper funtion that performs various methods including
/// white and yellow color detection, masking, ROI, and hough transform
/// </summary>
/// Preconditions:		Current image frame should be passed in
/// Postconditions:		RGB image with detected lanes
/// <param name="input_frame"></param>
/// <returns></returns>
cv::Mat FeatureExtractor::lane_detect(cv::Mat& input_frame) {
	// Get RGB img
	cv::Mat rgb_im = input_frame.clone();
	cv::Mat edges = this->extract_lane_colors(input_frame);

	// From the edges, remove unnecessary edges
	// propose a region of interest
	cv::Mat roi_im = this->propose_roi(edges,
		0, 1,
		1, 1,
		0.518, 0.59,
		0.518, 0.59,
		false
	);

	rgb_im = this->get_lanes(roi_im, rgb_im);
	return rgb_im;
}

/// <summary>
/// Wrapper funtion that performs various methods including
/// white and yellow color detection, masking and edge detection
/// </summary>
/// Preconditions:		Current image frame should be passed in with config object
/// Postconditions:		Image with edges containing lanes
/// <param name="input_frame"></param>
/// <returns></returns>
cv::Mat FeatureExtractor::extract_lane_colors(ConfigurationParameters& config, cv::Mat& input_frame) {
	cv::Mat hls_im;
	cv::cvtColor(input_frame, hls_im, cv::COLOR_BGR2HLS);

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
	cv::GaussianBlur(combined, blurred, { config.smoothing_kernel_size,config.smoothing_kernel_size }, 0);

	// Edge detect
	cv::Mat edges;
	cv::Canny(blurred, edges, config.canny_thresh1, config.canny_thresh2);
	return edges;
}

/// <summary>
/// Wrapper funtion that performs various methods including
/// white and yellow color detection, masking and edge detection
/// </summary>
/// Preconditions:		Current image frame should be passed in
/// Postconditions:		Image with edges containing lanes
/// <param name="input_frame"></param>
/// <returns></returns>
cv::Mat FeatureExtractor::extract_lane_colors(cv::Mat& input_frame) {
	cv::Mat hls_im;
	cv::cvtColor(input_frame, hls_im, cv::COLOR_BGR2HLS);

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
	return edges;
}

/// <summary>
/// Displays the new image
/// Pre: Image that has gone through lane detection
/// Post: Image is displayed
/// </summary>
/// <param name="img"></param>
/// <param name="x_window"></param>
/// <param name="y_window"></param>
/// <param name="delay"></param>
void FeatureExtractor::show_image(cv::Mat& img, int x_window, int y_window, int delay) {
	cv::namedWindow("output", cv::WINDOW_NORMAL);
	cv::resizeWindow("output", img.cols / x_window, img.rows / y_window);
	cv::imshow("output", img);
	cv::waitKey(delay);
}

/// <summary>
/// Loads the dataset for feature extraction
/// Pre: Has a dataset
/// Post: Data is now in vector
/// </summary>
/// <param name="car_ds_loc">Name of DS</param>
/// <param name="non_car_ds">Name of DS</param>
/// <param name="debug"></param>
/// <param name="num_imgs">Num of images</param>
/// <returns>pair of vector mats</returns>
std::pair<std::vector<cv::Mat>, std::vector<int>> FeatureExtractor::load_dataset(std::string car_ds_loc, std::string non_car_ds, bool debug, int num_imgs) {

	std::vector<cv::Mat> vehicles_arr = this->load_images(car_ds_loc, debug, num_imgs);
	std::vector<int> vehicle_labels(vehicles_arr.size(), 1);

	std::vector<cv::Mat> non_vehicles_arr = this->load_images(non_car_ds, debug, num_imgs);
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

/// <summary>
/// 
/// </summary>
/// <param name="dataset_loc"></param>
/// <param name="label"></param>
/// <param name="debug"></param>
/// <param name="num_imgs"></param>
/// <returns></returns>
std::pair<std::vector<cv::Mat>, std::vector<int>> FeatureExtractor::load_dataset(std::string dataset_loc, int label, bool debug, int num_imgs) {
	std::vector<cv::Mat> x_data = this->load_images(dataset_loc, debug, num_imgs);
	std::vector<int> y_data(x_data.size(), label);
	std::pair<std::vector<cv::Mat>, std::vector<int>> dataset = { x_data,y_data };
	return dataset;
}

/// <summary>
/// Loads images
/// </summary>
/// <param name="dataset_loc"></param>
/// <param name="debug"></param>
/// <param name="num_imgs"></param>
/// <returns></returns>
std::vector<cv::Mat> FeatureExtractor::load_images(std::string dataset_loc, bool debug, int num_imgs) {
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
				if (debug)
					count++;
			}
		}
	}
	return images;
}

/// <summary>
/// featurize_dataset
/// Calculates the histogram of gradients that will be used 
/// as features for the SVM model
/// Preconditions:	Need to have the config struct passed in along with the
///					dataset which is a vector of Mat images
/// Postconditions:	Vector of HOG features for each image is returned
/// </summary>
/// <param name="dataset"></param>
/// <param name="debug"></param>
/// <returns></returns>
std::vector<cv::Mat> FeatureExtractor::featurize_dataset(ConfigurationParameters& config, std::vector<cv::Mat>& dataset, bool debug) {
	// HOG Params
	cv::Size window_stride = { config.win_stride,config.win_stride };
	cv::Size padding = { 0,0 };
	std::vector<float> descriptors;
	std::vector<cv::Point> location_pts;
	cv::HOGDescriptor hog;
	hog.winSize = cv::Size(config.window_size, config.window_size);

	std::vector<cv::Mat> hog_ims;
	for (int i = 0; i < dataset.size(); i++) {

		cv::Mat curr_im = dataset[i];
		cv::Mat gray_im;
		cv::cvtColor(curr_im, gray_im, cv::COLOR_BGR2GRAY);

		//Histogram of oriented gradients
		hog.compute(gray_im, descriptors, window_stride, padding, location_pts);
		int des_sz = descriptors.size();

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

/// <summary>
/// split
/// This function splits a string based on a specified delimiter
/// Preconditions:	specify the string in question and the char delimiter 
/// Postconditions:	returns a string vector of items parsed by the delimiter
/// </summary>
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

/// <summary>
/// get_name_num
/// This function just gets the numeral name of the image
/// Preconditions:		string directory path to the image is given
/// Postconditions:		string number is returned
/// </summary>
/// <param name="file_loc"></param>
/// <returns></returns>
std::string FeatureExtractor::get_name_num(std::string& file_loc) {
	std::vector<std::string> f_path_items = this->split(file_loc, '/');
	std::string f_name = f_path_items[f_path_items.size() - 1];
	std::vector<std::string> f_name_items = this->split(f_name, '.');
	std::string f_name_num = f_name_items[0];
	return f_name_num;
}

/// <summary>
/// Normalize the dataset
/// Pre: Has a dataset to normalzie
/// Post: Returns a normalized dataset
/// </summary>
/// <param name="x_data"></param>
/// <returns></returns>
cv::Mat FeatureExtractor::normalize_dataset(cv::Mat& x_data) {
	cv::Mat norm_x_data;
	cv::normalize(x_data, norm_x_data, 1.0, 0.0, cv::NORM_INF);
	return norm_x_data;
}

/// <summary>
/// Converts the vector of matrices into a matrix of matrices
/// Preconditions:	vector of matrix features and vector of labels must 
///					be passed in
/// Postconditions:	matrix of matrices of hog features is returned
/// </summary>
/// <param name="x_data"></param>
/// <param name="y_data"></param>
/// https://github.com/opencv/opencv/blob/master/samples/cpp/train_HOG.cpp
/// <returns></returns>
std::pair<cv::Mat, cv::Mat> FeatureExtractor::prepare_training_data(std::vector<cv::Mat>& x_data, std::vector<int>& y_data) {
	// Convert x_data to mat
	int rs = x_data.size();
	int cs = std::max(x_data[0].rows, x_data[0].cols);
	cv::Mat x_data_mat(rs, cs, CV_32FC1);
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
	std::pair<cv::Mat, cv::Mat> ret = { x_data_mat,cv::Mat(y_data) };
	return ret;
}

/// <summary>
/// [DEPRACATED/UNUSED/UNTESTED]
/// </summary>
/// <param name="img"></param>
/// <param name="win_stride"></param>
/// <param name="window_size"></param>
/// <param name="scale"></param>
/// <param name="model"></param>
/// <returns></returns>
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

		for (int row = 0; row + win_stride.height <= gray_im.rows - win_stride.height; row += win_stride.height) {
			for (int col = 0; col + win_stride.width <= gray_im.cols - win_stride.width; col += win_stride.width) {
				cv::Rect curr_window(col, row, window_size.width, window_size.height);
				if (curr_window.x >= 0 && curr_window.y >= 0 && curr_window.width + curr_window.x < img.cols && curr_window.height + curr_window.y < img.rows) {
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
