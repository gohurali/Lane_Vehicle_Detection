#include "feature_descriptor.h"

void show_image(cv::Mat& img, int x_window, int y_window, int delay) {
	cv::namedWindow("output", cv::WINDOW_NORMAL);
	cv::resizeWindow("output", img.cols / x_window, img.rows / y_window);
	cv::imshow("output", img);
	cv::waitKey(delay);
}

int main() {
	bool debug = true;

	cv::Mat input = cv::imread("images/input.jpg");
	if (debug)
		printf("Input img size (r,c) %i x %i\n",input.rows,input.cols);
		show_image(input, 1, 1, 5000);

	return 0;
}