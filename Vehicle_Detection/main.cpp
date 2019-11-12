#include "feature_extractor.h"

int main() {
	bool debug = false;
	FeatureExtractor fe;
	cv::Mat input = cv::imread("images/lanes_1.jpg");
	cv::Mat output = fe.lane_detect(input);
	fe.show_image(output, 1, 1, 5000);
	return 0;
}