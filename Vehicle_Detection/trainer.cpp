#include "trainer.h"

/// <summary>
/// Train support vector machine model for classification
/// on x and y dataset
/// </summary>
/// Preconditions:		x and y data must be formatted as matrices and specified model_name
/// Postconditions:		model is saved in the current project directory
/// <param name="x_data"></param>
/// <param name="y_data"></param>
/// <param name="model_fname"></param>
void Trainer::train_svm(cv::Mat& x_data, cv::Mat& y_data, std::string model_fname) {
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
	svm_model->setType(cv::ml::SVM::C_SVC);
	svm_model->setKernel(cv::ml::SVM::LINEAR);
	svm_model->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-3));
	svm_model->train(x_data, cv::ml::ROW_SAMPLE, y_data);
	svm_model->save(model_fname);
	printf("-- Training Complete -- \n");
}

/// <summary>
/// train_test_svm
/// Debug method where the split dataset is used
/// model is then tested with the test set to see
/// the model performance
/// Preconditions:		The dataset must be split into train and test sets, additionally the user
///						can specify if would like to serialize the model
///	Postconditions:		Returns the model's performance and serializes the model if specified by user
/// </summary>
/// <param name="x_train"></param>
/// <param name="y_train"></param>
/// <param name="x_test"></param>
/// <param name="y_test"></param>
/// <param name="save_model"></param>
/// <param name="model_fname"></param>
void Trainer::train_test_svm(const cv::Mat& x_train, const cv::Mat& y_train,const cv::Mat& x_test, const cv::Mat& y_test, bool save_model, std::string model_fname) {
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
	svm_model->setType(cv::ml::SVM::C_SVC);
	svm_model->setKernel(cv::ml::SVM::LINEAR);
	svm_model->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-3));
	svm_model->train(x_train, cv::ml::ROW_SAMPLE, y_train);
	if (save_model) {
		svm_model->save(model_fname);
	}
	printf("-- Training Complete -- \n");

	cv::Mat predictions;
	svm_model->predict(x_test, predictions);

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
	printf("SVM Test Accuracy = %f\n", score);
}

/// <summary>
/// train_test_split
/// Preconditions:		x_data and y_data and a defined test set size are needed to be passed in
/// Postconditions:		Pointer to TrainData dataset is returned to be able to get training and test data
/// </summary>
/// <param name="x_data"></param>
/// <param name="y_data"></param>
/// <param name="test_size"></param>
/// <returns></returns>
cv::Ptr<cv::ml::TrainData> Trainer::train_test_split(cv::Mat& x_data, cv::Mat& y_data, int test_size) {
	int train_size = x_data.rows - test_size;
	cv::Ptr<cv::ml::TrainData> dataset = cv::ml::TrainData::create(x_data, cv::ml::ROW_SAMPLE, y_data);
	dataset->setTrainTestSplit(train_size);
	return dataset;
}