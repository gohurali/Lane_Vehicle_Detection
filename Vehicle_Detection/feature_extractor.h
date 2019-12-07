/// FeatureExtractor
/// By Gohur, Umair, Will
/// Extracts features of input images. These features include both
/// cars and lane lines. Lane lines are detected using Hough Transform
/// and are highlighted (traced) with a red line and filled in between. 
/// Cars features are loaded. 
/// Pre: Input images of cars and lane lines
/// Post: Output of lane lines and cars with bound boxes.
#ifndef FEATURE_DESCRIPTOR_H
#define	FEATURE_DESCRIPTOR_H
#define _USE_MATH_DEFINES
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/ml.hpp>
#include <filesystem>
#include <vector>
#include <sstream>
#include <iostream>
#include <string>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "config.h"

class FeatureExtractor {
public:
	
	/// Creates a mask given lower and upper bound limitations
	/// Preconditions:	Mat img must be an image
	/// 			vector lower_b must be a vector of lower bounds
	/// 			vector upper_b must be a vector of upper bounds
	/// Postconditions:	Returns the mask that is derived from the input image and the specified bounds
	cv::Mat mask_color(cv::Mat&, std::vector<int>&,std::vector<int>&);
	
	/// combine_mask  creates a combined bit-wise disjunction of mask1 and mask2
	/// Pre:
	/// Mask1 a mat that is a mask
	/// Mask2 a mat that is a mask
	/// Post:
	/// combined is the bit-wise disjunction of mask1 and 2.
	cv::Mat combine_mask(cv::Mat&, cv::Mat&);

	/// remove_middle_polygons
	/// the goal of this method is to iterate through
	/// the mask and find the pixels within the bounds of 
	/// the mask and set them to zero
	cv::Mat remove_middle_polygons(
		cv::Mat&, 
		cv::Mat&
	);

	/// Creates inner cover mask
	cv::Mat create_inner_cover_mask(
		cv::Mat&,
		std::vector<std::pair<float, float>>&,
		bool debug = (false)
	);

	/// Propose the ROI, the ROI allows for our program to compute faster by specifying a certain region
	/// to focus our search on. 
	/// Pre: Need an image
	/// Post: Returns the region of interest in image
	cv::Mat propose_roi(
		cv::Mat&, 
		std::vector<std::pair<float, float>>&,
		bool debug = (false)
	);
	
	/// Proposes the ROI using pre-defined region selection
	/// Pre: Input image and specified locations
	/// Post: Outputs the specified region of interest
	cv::Mat propose_roi(
		cv::Mat&,
		double, double,
		double, double,
		double, double,
		double, double,
		bool debug = (false)
	);

	/// Lane Detect
	/// Wrapper funtion that performs various methods including
	/// white and yellow color detection, masking, ROI, and hough transform
	/// Config based lane detection
	/// lanes are detected based on the parameters
	/// specified in the config.h config file.
	/// Preconditions:		configuration object should be passed in along with the
	///						current image frame
	/// Postconditions:		RGB image with detected lanes
	cv::Mat lane_detect(
		ConfigurationParameters& config,
		cv::Mat&
	);
	
	/// Wrapper funtion that performs various methods including
	/// white and yellow color detection, masking, ROI, and hough transform
	/// Preconditions:		Current image frame should be passed in
	/// Postconditions:		RGB image with detected lanes
	cv::Mat lane_detect(
		cv::Mat&
	);
	
	/// Wrapper funtion that performs various methods including
	/// white and yellow color detection, masking, ROI, and hough transform
	/// Preconditions:		Current image frame and ROI of lanes should be determined and passed in
	/// Postconditions:		RGB image with detected lanes
	cv::Mat lane_detect(
		cv::Mat&, 
		std::vector<std::pair<float, float>>&
	);

	/// Wrapper funtion that performs various methods including
	/// white and yellow color detection, masking and edge detection
	/// Preconditions:		Current image frame should be passed in
	/// Postconditions:		Image with edges containing lanes
	cv::Mat extract_lane_colors(cv::Mat&);
	
	/// Wrapper funtion that performs various methods including
	/// white and yellow color detection, masking and edge detection
	/// Preconditions:		Current image frame should be passed in with config object
	/// Postconditions:		Image with edges containing lanes
	cv::Mat extract_lane_colors(
		ConfigurationParameters& config,
		cv::Mat&
	);

	/// gets lanes with predefined threshold params
	/// Preconditions:		input and output image should be given
	/// Postconditions:		img with detected lane lines displayed
	cv::Mat get_lanes(cv::Mat&, cv::Mat&);
	
	/// Get's lane lines using Hough Transform
	/// Pre: Accepts image input and thresholds
	/// Post: Image with lane lines that were detected with hough transform 
	cv::Mat get_lanes(cv::Mat&, cv::Mat&, int, int);
	
	/// Performs a Hough Transform and obtains the lane lines
	/// A connection line is calculated to see the extent of the
	/// detection. Additionally, the lanes are extrapolated
	/// Preconditions: Input image and specified locations
	/// Postconditions: Outputs the specified region of interest
	cv::Mat get_lanes(
		ConfigurationParameters& config,
		cv::Mat&, 
		cv::Mat&, 
		int, 
		int
	);

	/// Draws the right and left lanes based on the found lines
	/// Preconditions:	The highest and lowest points must have been found and the
	///					shorter lane should be extrapolated to take sure lane lines
	///					are equivalent
	/// Postconditions:	Lanes are draws on the output image
	void draw_lane_lines(
		cv::Mat& output,
		cv::Vec4i& min_points,
		cv::Point& adjusted_right_min,
		cv::Vec4i& top_line
	);
	
	/// Draws an overlay between the lane lines
	/// Preconditions:	The highest and lowest points must have been found and the
	///					shorter lane should be extrapolated to take sure lane lines
	///					are equivalent
	/// Postconditions:	An partially transparent overlay is drawn on the img
	void draw_lane_overlay(
		cv::Mat& output,
		cv::Vec4i& min_points,
		cv::Point& adjusted_right_min,
		cv::Vec4i& top_line
	);

	/// find_lowest_point
	/// Preconditions:	Vector of 4 value vectors representing hough lines must be
	///					provided. This is pre-calculated in the get_lanes method.
	///					Addtionally, a middle_pt threshold must be provided to represent
	///					the location of the middle of the lane
	/// Postconditions:	Returns a 4 point vector containing the location of the lowest point
	///					of each of the lanes
	cv::Vec4i find_lowest_point(
		std::vector<cv::Vec4i>&,
		int middle_pt = (650),
		bool use_middle = (true)
	);
	
	/// find_highest_point
	/// Preconditions:	Vector of 4 value vectors representing hough lines must be
	///					provided. This is pre-calculated in the get_lanes method.
	///					Addtionally, a middle_pt threshold must be provided to represent
	///					the location of the middle of the lane
	/// Postconditions:	Returns a 4 point vector containing the location of the highest point
	///					of each of the lanes
	cv::Vec4i find_highest_point(
		std::vector<cv::Vec4i>&, 
		int middle_pt = (650),
		bool use_middle = (true)
	);
	
	/// find_lowest_point
	/// Preconditions:	Vector of 4 value vectors representing hough lines must be
	///					provided. This is pre-calculated in the get_lanes method.
	///					Addtionally, a middle_pt threshold must be provided to represent
	///					the location of the middle of the lane
	/// Postconditions:	Returns a 4 point vector containing the location of the lowest point
	///					of each of the lanes
	cv::Vec4i find_lowest_point(
		std::vector<cv::Vec4i>&, 
		int l_threshold = (650),
		int r_threshold = (650)
	);
	
	/// find_highest_point
	/// Preconditions:	Vector of 4 value vectors representing hough lines must be
	///					provided. This is pre-calculated in the get_lanes method.
	///					Addtionally, a middle_pt threshold must be provided to represent
	///					the location of the middle of the lane
	/// Postconditions:	Returns a 4 point vector containing the location of the highest point
	///					of each of the lanes
	cv::Vec4i find_highest_point(
		std::vector<cv::Vec4i>&, 
		int l_threshold = (650),
		int r_threshold = (650)
	);

	/// extrapolate_line
	/// extends a linear function by a given y point
	/// Preconditions: A cv::Vec4 integer line and a y_pt
	/// Postconditions: Return the new point with the updated x_pt
	cv::Point extrapolate_line(cv::Vec4i&, int);
	
	/// Displays the new image
	/// Preconditions: 	Image that has gone through lane detection
	/// Postconditions: 	Image is displayed
	void show_image(cv::Mat&,int,int,int);

	/// featurize_dataset
	/// Calculates the histogram of gradients that will be used 
	/// as features for the SVM model
	/// Preconditions:	Need to have the config struct passed in along with the
	///			dataset which is a vector of Mat images
	/// Postconditions:	Vector of HOG features for each image is returned
	std::vector<cv::Mat> featurize_dataset(
		ConfigurationParameters& config,
		std::vector<cv::Mat>&,
		bool
	);

	/// Normalize the dataset
	/// Preconditions: 	Has a dataset to normalzie
	/// Postconditions: 	Returns a normalized dataset
	cv::Mat normalize_dataset(cv::Mat&);

	/// Converts the vector of matrices into a matrix of matrices
	/// Preconditions:	vector of matrix features and vector of labels must 
	///			be passed in
	/// Postconditions:	matrix of matrices of hog features is returned
	std::pair<cv::Mat, cv::Mat> prepare_training_data(std::vector<cv::Mat>&,std::vector<int>&);

	/// [DEPRACATED/UNUSED/UNTESTED]
	std::vector<cv::Rect> sliding_window(
		cv::Mat& img, 
		cv::Size& win_stride, 
		cv::Size& window_size, 
		float scale,
		const cv::Ptr <cv::ml::SVM>& model
	);

	/// Loads the dataset for feature extraction
	/// Preconditions: 	Has a dataset
	/// Postconditions: 	Data is now in vector
	std::pair<std::vector<cv::Mat>, std::vector<int>> load_dataset(
		std::string, 
		std::string, 
		bool debug,
		int num_imgs = (200)
	);
	
	/// Loads the dataset for feature extraction
	/// Preconditions: 	Has a dataset
	/// Postconditions: 	Data is now in vector
	std::pair<std::vector<cv::Mat>, std::vector<int>> load_dataset(
		std::string, 
		int, 
		bool,
		int num_imgs = (200)
	);

	/// split
	/// This function splits a string based on a specified delimiter
	/// Preconditions:	specify the string in question and the char delimiter 
	/// Postconditions:	returns a string vector of items parsed by the delimiter
	std::vector<std::string> split(const std::string&, char);
	
	/// get_name_num
	/// This function just gets the numeral name of the image
	/// Preconditions:		string directory path to the image is given
	/// Postconditions:		string number is returned
	std::string get_name_num(std::string&);
	
private:
	
	/// Loads images
	std::vector<cv::Mat> load_images(std::string,bool,int num_imgs = (200));
};

#endif
