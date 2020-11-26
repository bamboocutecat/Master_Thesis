#include <opencv2/opencv.hpp>  
#include <iostream>  

using namespace std;  

int main()  
{  
	cout << "测试两个摄像头同时读取数据" << endl;  

	//读取内部参数
	CvMat *Intrinsics_Camera_Left = (CvMat *)cvLoad("Intrinsics_Camera_Left.xml");
	CvMat *Intrinsics_Camera_Right = (CvMat *)cvLoad("Intrinsics_Camera_Right.xml");
	CvMat *Distortion_Camera_Left = (CvMat *)cvLoad("Distortion_Camera_Left.xml");
        CvMat *Distortion_Camera_Right = (CvMat *)cvLoad("Distortion_Camera_Right.xml");
	CvMat *Translation_matlab = (CvMat *)cvLoad("Translation.xml");
	CvMat *RotRodrigues_matlab = (CvMat *)cvLoad("RotRodrigues.xml");
	CvMat *R_opencv = cvCreateMat(3, 3, CV_64F);
	cvRodrigues2(RotRodrigues_matlab, R_opencv);

	//创建映射阵
	IplImage *Left_Mapx = cvCreateImage(cvSize(640,480), IPL_DEPTH_32F, 1);
	IplImage *Left_Mapy = cvCreateImage(cvSize(640,480), IPL_DEPTH_32F, 1);
	IplImage *Right_Mapx = cvCreateImage(cvSize(640,480), IPL_DEPTH_32F, 1);
	IplImage *Right_Mapy = cvCreateImage(cvSize(640,480), IPL_DEPTH_32F, 1);
	CvMat *Rl = cvCreateMat(3, 3, CV_64F);
	CvMat *Rr = cvCreateMat(3, 3, CV_64F);
	CvMat *Pl = cvCreateMat(3, 4, CV_64F);
	CvMat *Pr = cvCreateMat(3, 4, CV_64F);
	//cvStereoRectify(Intrinsics_Camera_Left, Intrinsics_Camera_Right, \
		            Distortion_Camera_Left, Distortion_Camera_Right, \
					cvSize(640, 480), R_opencv, Translation_matlab,  \
					Rl, Rr, Pl, Pr);
	cvStereoRectify(Intrinsics_Camera_Left, Intrinsics_Camera_Right, \
		Distortion_Camera_Left, Distortion_Camera_Right, \
		cvSize(640, 480), R_opencv, Translation_matlab,  \
		Rl, Rr, Pl, Pr, 0, 1024, 0);//增加图像缩放，去除死区
	cvInitUndistortRectifyMap(Intrinsics_Camera_Left, Distortion_Camera_Left, Rl, Pl, \
		                      Left_Mapx, Left_Mapy);
	cvInitUndistortRectifyMap(Intrinsics_Camera_Right, Distortion_Camera_Right, Rr, Pr, \
		                      Right_Mapx, Right_Mapy);

	CvCapture* cap_left;  
	CvCapture* cap_right;  

	IplImage *img0 , *img1;  
	IplImage *img_left, *img_right;  
	IplImage *img_left_Change, *img_right_Change;  
	cvNamedWindow("camera_left");  
	cvNamedWindow("camera_right");  

	cap_left = cvCreateCameraCapture(1);  
	assert(cap_left != NULL);  
	cap_right = cvCreateCameraCapture(0);  
	assert(cap_right != NULL);  

	while(1)  
	{  
		cvGrabFrame(cap_left);
		cvGrabFrame(cap_right);

		img0 = cvRetrieveFrame(cap_left);
		img1 = cvRetrieveFrame(cap_right);

		if( !img0 || !img1)
		{  
			cout << "camera0 error" << endl;  
			break;  
		}  

		img_left = cvCloneImage(img0);
		img_right = cvCloneImage(img1);
		img_left_Change = cvCloneImage(img0);
		img_right_Change = cvCloneImage(img1);
		cvRemap(img_left, img_left_Change, Left_Mapx, Left_Mapy);
		cvRemap(img_right, img_right_Change, Right_Mapx, Right_Mapy);

		cvLine(img_left_Change, cvPoint(0,48), cvPoint(640-1, 48), cvScalar(255, 0, 0));
		cvLine(img_left_Change, cvPoint(0,48*2), cvPoint(640-1, 48*2), cvScalar(255, 0, 0));
		cvLine(img_left_Change, cvPoint(0,48*3), cvPoint(640-1, 48*3), cvScalar(255, 0, 0));
		cvLine(img_left_Change, cvPoint(0,48*4), cvPoint(640-1, 48*4), cvScalar(255, 0, 0));
		cvLine(img_left_Change, cvPoint(0,48*5), cvPoint(640-1, 48*5), cvScalar(255, 0, 0));
		cvLine(img_left_Change, cvPoint(0,48*6), cvPoint(640-1, 48*6), cvScalar(255, 0, 0));
		cvLine(img_left_Change, cvPoint(0,48*7), cvPoint(640-1, 48*7), cvScalar(255, 0, 0));
		cvLine(img_left_Change, cvPoint(0,48*8), cvPoint(640-1, 48*8), cvScalar(255, 0, 0));
		cvLine(img_left_Change, cvPoint(0,48*9), cvPoint(640-1, 48*9), cvScalar(255, 0, 0));
		cvLine(img_right_Change, cvPoint(0,48), cvPoint(640-1, 48), cvScalar(255, 0, 0));
		cvLine(img_right_Change, cvPoint(0,48*2), cvPoint(640-1, 48*2), cvScalar(255, 0, 0));
		cvLine(img_right_Change, cvPoint(0,48*3), cvPoint(640-1, 48*3), cvScalar(255, 0, 0));
		cvLine(img_right_Change, cvPoint(0,48*4), cvPoint(640-1, 48*4), cvScalar(255, 0, 0));
		cvLine(img_right_Change, cvPoint(0,48*5), cvPoint(640-1, 48*5), cvScalar(255, 0, 0));
		cvLine(img_right_Change, cvPoint(0,48*6), cvPoint(640-1, 48*6), cvScalar(255, 0, 0));
		cvLine(img_right_Change, cvPoint(0,48*7), cvPoint(640-1, 48*7), cvScalar(255, 0, 0));
		cvLine(img_right_Change, cvPoint(0,48*8), cvPoint(640-1, 48*8), cvScalar(255, 0, 0));
		cvLine(img_right_Change, cvPoint(0,48*9), cvPoint(640-1, 48*9), cvScalar(255, 0, 0));

		cvShowImage("camera_left", img_left);  
		cvShowImage("camera_right", img_right);  
		cvShowImage("camera_left_Change", img_left_Change);  
		cvShowImage("camera_right_Change", img_right_Change);  

		char c = cvWaitKey(33);  
		if(c == 27)   
			break;  

		cvReleaseImage(&img_left);
		cvReleaseImage(&img_right);
		cvReleaseImage(&img_left_Change);
		cvReleaseImage(&img_right_Change);

	}  

	cvReleaseCapture(&cap_left);  
	cvReleaseCapture(&cap_right);  
	cvDestroyWindow("camera_left");  
	cvDestroyWindow("camera_right");  
 
	return 0;  

}  