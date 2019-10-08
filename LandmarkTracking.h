#ifndef ZEUSEESFACETRACKING_H
#define ZEUSEESFACETRACKING_H
#include <opencv2/opencv.hpp>
//#include <thread>
#include "mtcnn.h"
#include "time.h"

namespace Shape {

	template <typename T> class Rect {
	public:
		Rect() {}
		Rect(T x, T y, T w, T h) {
			this->x = x;
			this->y = y;
			this->width = w;
			height = h;

		}
		T x;
		T y;
		T width;
		T height;

		cv::Rect convert_cv_rect(int _height, int _width)
		{
			cv::Rect Rect_(static_cast<int>(x * _width), static_cast<int>(y * _height),
				static_cast<int>(width * _width), static_cast<int>(height * _height));
			return Rect_;
		}
	};
}



cv::Rect boundingRect(const std::vector<cv::Point>& pts) {
	if (pts.size() > 1)
	{
		int xmin = pts[0].x;
		int ymin = pts[0].y;
		int xmax = pts[0].x;
		int ymax = pts[0].y;
		for (int i = 1; i < pts.size(); i++)
		{
			if (pts[i].x < xmin)
				xmin = pts[i].x;
			if (pts[i].y < ymin)
				ymin = pts[i].y;
			if (pts[i].x > xmax)
				xmax = pts[i].x;
			if (pts[i].y > ymax)
				ymax = pts[i].y;
		}
		return cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);
	}
}


//typedef int T;
//T i = 1;


class Face {
public:

	Face(int instance_id, Shape::Rect<float> rect) {
		face_id = instance_id;
		landmark = std::shared_ptr<vector<cv::Point> >(new vector<cv::Point>(5));

		face_location = rect;
		isCanShow = false; //追踪一次后待框稳定后即可显示
		for (int i = 0; i < 5; i++)
		{
			(*landmark)[i].x = -1;
			(*landmark)[i].y = -1;
		}
	}

	Face() {

		landmark = std::shared_ptr<vector<cv::Point> >(new vector<cv::Point>(5));
		isCanShow = false; //追踪一次后待框稳定后即可显示
	}

	std::shared_ptr<vector<cv::Point> > landmark;

	int face_id = -1;
	long frameId = 0;
	int ptr_num = 0;

	Shape::Rect<float> face_location;
	bool isCanShow;
	cv::Mat frame_face_prev;

	static cv::Rect SquarePadding(cv::Rect facebox, int margin_rows, int margin_cols, bool max)
	{
		int c_x = facebox.x + facebox.width / 2;
		int c_y = facebox.y + facebox.height / 2;
		int large = 0;
		if (max)
			large = max(facebox.height, facebox.width) / 2;
		else
			large = min(facebox.height, facebox.width) / 2;
		cv::Rect rectNot(c_x - large, c_y - large, c_x + large, c_y + large);
		rectNot.x = max(0, rectNot.x);
		rectNot.y = max(0, rectNot.y);
		rectNot.height = min(rectNot.height, margin_rows - 1);
		rectNot.width = min(rectNot.width, margin_cols - 1);
		if (rectNot.height - rectNot.y != rectNot.width - rectNot.x)
			return SquarePadding(cv::Rect(rectNot.x, rectNot.y, rectNot.width - rectNot.x, rectNot.height - rectNot.y), margin_rows, margin_cols, false);

		return cv::Rect(rectNot.x, rectNot.y, rectNot.width - rectNot.x, rectNot.height - rectNot.y);
	}

	static cv::Rect SquarePadding(cv::Rect facebox, int padding)
	{

		int c_x = facebox.x - padding;
		int c_y = facebox.y - padding;
		return cv::Rect(facebox.x - padding, facebox.y - padding, facebox.width + padding * 2, facebox.height + padding * 2);;
	}

	static double getDistance(cv::Point x, cv::Point y)
	{
		return sqrt((x.x - y.x) * (x.x - y.x) + (x.y - y.y) * (x.y - y.y));
	}


	vector<vector<cv::Point> > faceSequence;
	vector<vector<float>> attitudeSequence;


};


float r_iou(Face & f1, Face & f2)
{
	float IOU = 0;
	float maxX = 0;
	float maxY = 0;
	float minX = 0;
	float minY = 0;
	cv::Rect b1 = cv::boundingRect(*f1.landmark);
	cv::Rect b2 = cv::boundingRect(*f2.landmark);
	maxX = max(b1.x, b2.x);
	maxY = max(b1.y, b2.y);
	minX = min(b1.x + b1.width, b2.x + b2.width);
	minY = min(b1.y + b1.height, b2.y + b2.height);
	//maxX1 and maxY1 reuse
	maxX = ((minX - maxX + 1)>0) ? (minX - maxX + 1) : 0;
	maxY = ((minY - maxY + 1)>0) ? (minY - maxY + 1) : 0;
	IOU = maxX * maxY;
	IOU = IOU / (b1.area() + b2.area() - IOU);

	return IOU;
}

//稳定框 暂时有bug
void SmoothBbox(std::vector<Face>& finalBbox)
{
	static std::vector<Face> preBbox_;
	for (int i = 0; i < finalBbox.size(); i++) {
		for (int j = 0; j < preBbox_.size(); j++) {
			if (r_iou(finalBbox[i], preBbox_[j]) > 0.90)
			{
				finalBbox[i] = preBbox_[j];
			}
			else if (r_iou(finalBbox[i], preBbox_[j]) > 0.6) {
				(*finalBbox[i].landmark)[0].x = ((*finalBbox[i].landmark)[0].x + (*preBbox_[i].landmark)[0].x) / 2;
				(*finalBbox[i].landmark)[0].y = ((*finalBbox[i].landmark)[0].y + (*preBbox_[i].landmark)[0].y) / 2;
				(*finalBbox[i].landmark)[1].x = ((*finalBbox[i].landmark)[1].x + (*preBbox_[i].landmark)[1].x) / 2;
				(*finalBbox[i].landmark)[1].y = ((*finalBbox[i].landmark)[1].y + (*preBbox_[i].landmark)[1].y) / 2;
				
			}
		}
	}
	preBbox_ = finalBbox;

}

class FaceTracking {
public:
	FaceTracking(string modelPath)
	{
		this->detector = new MTCNN(modelPath);
		downSimpilingFactor = 1;
		faceMinSize = 70;
		this->detector->SetMinFace(faceMinSize);
		detection_Time = -1;

	}

	~FaceTracking() {
		delete this->detector;

	}

	void detecting(cv::Mat* image) {
		ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image->data, ncnn::Mat::PIXEL_BGR2RGB, image->cols, image->rows);
		std::vector<Bbox> finalBbox;
		detector->detect(ncnn_img, finalBbox);
		const int num_box = finalBbox.size();
		std::vector<cv::Rect> bbox;
		bbox.resize(num_box);
		candidateFaces_lock = 1;
		for (int i = 0; i < num_box; i++) {
			bbox[i] = cv::Rect(finalBbox[i].x1, finalBbox[i].y1, finalBbox[i].x2 - finalBbox[i].x1 + 1,
				finalBbox[i].y2 - finalBbox[i].y1 + 1);
			bbox[i] = Face::SquarePadding(bbox[i], image->rows, image->cols, true);
			Shape::Rect<float> f_rect(bbox[i].x / static_cast<float>(image->cols),
				bbox[i].y / static_cast<float>(image->rows),
				bbox[i].width / static_cast<float>(image->cols),
				bbox[i].height / static_cast<float>(image->rows)
			);
			std::shared_ptr<Face> face(new Face(trackingID, f_rect));
			(*image)(bbox[i]).copyTo(face->frame_face_prev);

			trackingID = trackingID + 1;
			candidateFaces.push_back(*face);
		}
		candidateFaces_lock = 0;
	}

	void Init(cv::Mat& image) {
		ImageHighDP = image;
		cv::Size lowDpSize(ImageHighDP.cols / downSimpilingFactor, ImageHighDP.rows / downSimpilingFactor);
		cv::resize(image, ImageLowDP, lowDpSize);
		trackingID = 0;
		detection_Interval = 350; //detect faces every 200 ms
		detecting(&image);
		stabilization = false;
		UI_height = image.rows;
		UI_width = image.cols;
	}


	void doingLandmark_onet(cv::Mat& face, std::vector<cv::Point>& pts, int zeroadd_x, int zeroadd_y, int stable_state = 0) {
		ncnn::Mat in = ncnn::Mat::from_pixels_resize(face.data, ncnn::Mat::PIXEL_BGR, face.cols, face.rows, 48, 48);
		const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };
		const float norm_vals[3] = { 1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5 };
		in.substract_mean_normalize(mean_vals, norm_vals);
		ncnn::Extractor Onet = detector->Onet.create_extractor();
		Onet.set_num_threads(2);
		Onet.input("data", in);
		ncnn::Mat out;
		Onet.extract("conv6-2", out);


		for (int j = 0; j < 2; j++) {
			int x = 0;
			int y = 0;
			if (j == 0) {
				x = static_cast<int>(out[j * 2 + 0] * face.cols) + zeroadd_x;
				y = static_cast<int>(out[j * 2 + 1] * face.rows) + zeroadd_y;
			}
			else {
				x = static_cast<int>(out[j * 2 + 0] * face.cols) + face.cols + zeroadd_x;
				y = static_cast<int>(out[j * 2 + 1] * face.rows) + face.rows + zeroadd_y;
			}

			cv::Point p(x, y);
			pts[j] = p;
		}

		//for (int j = 0; j < 5; j++) {
		//    int x = 0 ;
		//    int y=  0 ;
		//     x = static_cast<int>(out[j  + 0] * face.cols) + zeroadd_x;
		//     y = static_cast<int>(out[j + 5] * face.rows) + zeroadd_y;
		//   // __android_log_print(ANDROID_LOG_ERROR,"landmark","landmark %d %d",x,y);
		//    cv::Point p(x, y);
		//    pts[j] = p;
		//}

	}


	void tracking_corrfilter(const cv::Mat& frame, const cv::Mat& model, cv::Rect& trackBox, float scale)
	{
		trackBox.x /= scale;
		trackBox.y /= scale;
		trackBox.height /= scale;
		trackBox.width /= scale;
		int zeroadd_x = 0;
		int zeroadd_y = 0;
		cv::Mat frame_;
		cv::Mat model_;
		cv::resize(frame, frame_, cv::Size(), 1 / scale, 1 / scale);
		cv::resize(model, model_, cv::Size(), 1 / scale, 1 / scale);
		cv::Mat gray;
		cvtColor(frame_, gray, cv::COLOR_RGB2GRAY);
		cv::Mat gray_model;
		cvtColor(model_, gray_model, cv::COLOR_RGB2GRAY);
		cv::Rect searchWindow;
		searchWindow.width = trackBox.width * 3;
		searchWindow.height = trackBox.height * 3;
		searchWindow.x = trackBox.x + trackBox.width * 0.5 - searchWindow.width * 0.5;
		searchWindow.y = trackBox.y + trackBox.height * 0.5 - searchWindow.height * 0.5;
		searchWindow &= cv::Rect(0, 0, frame_.cols, frame_.rows);
		cv::Mat similarity;
		matchTemplate(gray(searchWindow), gray_model, similarity, cv::TM_CCOEFF_NORMED);
		double mag_r;
		cv::Point point;
		minMaxLoc(similarity, 0, &mag_r, 0, &point);
		trackBox.x = point.x + searchWindow.x;
		trackBox.y = point.y + searchWindow.y;
		trackBox.x *= scale;
		trackBox.y *= scale;
		trackBox.height *= scale;
		trackBox.width *= scale;
	}

	bool tracking(cv::Mat& image, Face& face)
	{
		cv::Rect faceROI = face.face_location.convert_cv_rect(image.rows, image.cols);
		cv::Mat faceROI_Image;
		tracking_corrfilter(image, face.frame_face_prev, faceROI, 2);
		image(faceROI).copyTo(faceROI_Image);
		//clock_t start_time = clock();
		(*face.landmark).resize(2);
		doingLandmark_onet(faceROI_Image, *face.landmark, faceROI.x, faceROI.y, face.frameId > 1);

		//clock_t finish_time = clock();
		//double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;
		cv::Rect bdbox((*face.landmark)[0].x, (*face.landmark)[0].y, (*face.landmark)[1].x - (*face.landmark)[0].x, (*face.landmark)[1].y - (*face.landmark)[0].y);
		// cv::Rect bdbox = cv::boundingRect((*face.landmark));
		// bdbox = Face::SquarePadding(bdbox, static_cast<int>(bdbox.height*0.55));
		bdbox = Face::SquarePadding(bdbox, static_cast<int>(bdbox.height * -0.05));
		bdbox = Face::SquarePadding(bdbox, image.rows, image.cols, 1);
		Shape::Rect<float> boxfloat(bdbox.x / static_cast<float>(image.cols),
			bdbox.y / static_cast<float>(image.rows),
			bdbox.width / static_cast<float>(image.cols),
			bdbox.height / static_cast<float>(image.rows));

		//        face.face_location.height = boxfloat.height;
		//        face.face_location.width= boxfloat.width;
		//        face.face_location.x= (faceROI.x*0.5 +bdbox.x*0.5)/static_cast<float>(image.cols);
		//        face.face_location.y = (faceROI.y*0.5 + bdbox.y*0.5)/static_cast<float>(image.rows);

		face.face_location = boxfloat;
		faceROI = face.face_location.convert_cv_rect(image.rows, image.cols);

		image(faceROI).copyTo(face.frame_face_prev);
		face.frameId += 1;
		ncnn::Extractor Rnet = detector->Rnet.create_extractor();
		const float mean_vals[3] = { 127.5, 127.5, 127.5 };
		const float norm_vals[3] = { 0.0078125, 0.0078125, 0.0078125 };
		ncnn::Mat rnet_data = ncnn::Mat::from_pixels_resize(faceROI_Image.data, ncnn::Mat::PIXEL_BGR2RGB, faceROI_Image.cols, faceROI_Image.rows, 24, 24);
		rnet_data.substract_mean_normalize(mean_vals, norm_vals);
		Rnet.input("data", rnet_data);
		ncnn::Mat out_origin;
		Rnet.extract("prob1", out_origin);
		face.isCanShow = true;
		if (out_origin[1] > 0.1) {
			//stablize
			float diff_x = 0;
			float diff_y = 0;
			return true;
		}
		return false;

	}
	void setMask(cv::Mat& image, cv::Rect& rect_mask)
	{

		int height = image.rows;
		int width = image.cols;
		cv::Mat subImage = image(rect_mask);
		subImage.setTo(0);
	}

	void update(cv::Mat& image)
	{
		ImageHighDP = image;
		//std::cout << trackingFace.size() << std::endl;
		if (candidateFaces.size() > 0 && !candidateFaces_lock)
		{
			for (int i = 0; i < candidateFaces.size(); i++)
			{
				trackingFace.push_back(candidateFaces[i]);
			}
			candidateFaces.clear();
		}
		for (vector<Face>::iterator iter = trackingFace.begin(); iter != trackingFace.end();)
		{
			if (!tracking(image, *iter))
			{
				iter = trackingFace.erase(iter); //追踪失败 则删除此人脸
			}
			else {
				iter++;
			}
		}
		if(stabilization)
			SmoothBbox(trackingFace);

		if (detection_Time < 0)
		{
			detection_Time = (double)cv::getTickCount();
		}
		else {
			double diff = (double)(cv::getTickCount() - detection_Time) * 1000 / cv::getTickFrequency();
			if (diff > detection_Interval)
			{
				cv::Size lowDpSize(ImageHighDP.cols / downSimpilingFactor, ImageHighDP.rows / downSimpilingFactor);
				cv::resize(image, ImageLowDP, lowDpSize);
				//set Mask to protect the tracking face not to be detected.
				for (auto& face : trackingFace)
				{
					Shape::Rect<float> rect = face.face_location;
					cv::Rect rect1 = rect.convert_cv_rect(ImageLowDP.rows, ImageLowDP.cols);
					setMask(ImageLowDP, rect1);
				}
				detection_Time = (double)cv::getTickCount();
				// do detection in thread
				detecting(&ImageLowDP);
			}

		}
	}



	vector<Face> trackingFace; //跟踪中的人脸
	int UI_width;
	int UI_height;


private:

	int isLostDetection;
	int isTracking;
	int isDetection;
	cv::Mat ImageHighDP;
	cv::Mat ImageLowDP;
	int downSimpilingFactor;
	int faceMinSize;
	MTCNN* detector;
	vector<Face> candidateFaces; // 将检测到的人脸放入此列队 待跟踪的人脸
	bool candidateFaces_lock;
	double detection_Time;
	double detection_Interval;
	float stable_factor_stage1 = 0.2f;
	float stable_factor_stage2 = 2.0f;
	int trackingID;
	bool stabilization;


};
#endif //ZEUSEESFACETRACKING_H
