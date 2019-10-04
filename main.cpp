#include "LandmarkTracking.h"

int main()
{
	string model_path = "./models";
	FaceTracking faceTrack(model_path);
	cv::Mat frame;
	cv::VideoCapture capture(0);
	if (!capture.isOpened())
	{
		return -1;
	}

	int frameIndex = 0;
	vector<int> IDs;
	vector<cv::Scalar> Colors;
	cv::Scalar color;
	srand((unsigned int)time(0));//初始化种子为随机值
	for (;;) {
		if (!capture.read(frame))
		{
			break;
		}
		int q = cv::waitKey(1);
		if (q == 27) break;

		
		//cv::transpose(frame, frame);
		//cv::flip(frame, frame, -1);
		//cv::flip(frame, frame, 1);
		double t1 = (double)cv::getTickCount();
		if (frameIndex == 0)
		{
			faceTrack.Init(frame);
			frameIndex = 1;
		}
		else {
			faceTrack.update(frame);
		}
		printf("total %gms\n", ((double)cv::getTickCount() - t1) * 1000 / cv::getTickFrequency());
		printf("------------------\n");
		
		std::vector<Face> faceActions = faceTrack.trackingFace;
		for (int i = 0; i < faceActions.size(); i++)
		{
			const Face &info = faceActions[i];
			cv::Rect rect = cv::boundingRect(*info.landmark);
			//Shape::Rect<float> frect = info.face_location;
			bool isExist = false;
			for (int j = 0; j < IDs.size(); j++)
			{
				if (IDs[j] == info.face_id)
				{
					color = Colors[j];
					isExist = true;
					break;
				}
			}

			if (!isExist)
			{
				IDs.push_back(info.face_id);
				int r = rand() % 255 + 1;
				int g = rand() % 255 + 1;
				int b = rand() % 255 + 1;
				color = cv::Scalar(r, g, b);
				Colors.push_back(color);
			}

			rectangle(frame, rect, color, 2);
		}
		imshow("frame", frame);
	}
	IDs.clear();
	Colors.clear();
	capture.release();
	cv::destroyAllWindows();
	return 0;
}