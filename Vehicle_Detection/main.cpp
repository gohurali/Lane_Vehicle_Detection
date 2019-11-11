#include "feature_descriptor.h"
#include <iostream>
void show_image(cv::Mat& img, int x_window, int y_window, int delay) {
	cv::namedWindow("output", cv::WINDOW_NORMAL);
	cv::resizeWindow("output", img.cols / x_window, img.rows / y_window);
	cv::imshow("output", img);
	cv::waitKey(delay);
}

int main() {
	bool debug = false;

	cv::Mat input = cv::imread("images/lanes_1.jpg");
	if (debug) {
		//printf("Input img size (r,c) %i x %i\n", input.rows, input.cols);
		show_image(input, 1, 1, 5000);
	}
	cv::Mat grey_im;
	cv::cvtColor(input, grey_im, cv::COLOR_BGR2GRAY);
	//show_image(grey_im, 1, 1, 5000);

	cv::Mat hls_im;
	cv::cvtColor(input, hls_im, cv::COLOR_BGR2HLS);
	//show_image(hls_im, 1, 1, 5000);

	// Lane Color Masking
	// -- Yellow --
	cv::Mat yellow_mask;
	std::vector<int> y_lower_b = { 10, 0, 100 };
	std::vector<int> y_upper_b = { 40, 255, 255 };
	cv::Mat y_lower_mat(y_lower_b);
	cv::Mat y_upper_mat(y_upper_b);
	cv::inRange(hls_im,y_lower_mat, y_upper_mat,yellow_mask);
	std::cout << "y_lower_mat = " << std::endl << y_lower_mat << std::endl;
	show_image(yellow_mask, 1, 1, 5000);

	cv::Mat white_mask;
	std::vector<int> w_lower_b = { 0, 200, 0 };
	std::vector<int> w_upper_b = { 200, 255, 255 };
	cv::Mat w_lower_mat(w_lower_b);
	cv::Mat w_upper_mat(w_upper_b);
	cv::inRange(hls_im, w_lower_mat, w_upper_mat, white_mask);
	show_image(white_mask, 1, 1, 5000);

	// Combine
	cv::Mat combined;
	cv::bitwise_or(yellow_mask, white_mask,combined);
	show_image(combined, 1, 1, 5000);

	//Blur
	//cv::Mat blurred;
	//cv::GaussianBlur(hls_im, blurred, {15,15}, 3, 3);

	//cv::Mat edges;
	//cv::Canny(blurred, edges, 100,150);
	//show_image(edges, 1, 1, 5000);

	return 0;
}