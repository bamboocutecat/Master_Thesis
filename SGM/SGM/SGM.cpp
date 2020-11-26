#include <opencv2/core/version.hpp>
#if CV_MAJOR_VERSION >= 3
#include <opencv2/imgcodecs.hpp>
#else
#include <opencv2/highgui/highgui.hpp>
#endif
#include <opencv2/imgproc/imgproc.hpp>

#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/StereoDisparity.h>

#include <cstring> // for memset
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>

// Pybind11 block
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

cv::Size camera_size = (1280, 720);

namespace py = pybind11;

cv::Mat numpy_uint8_3c_to_cv_mat(py::array_t<unsigned char> &input)
{

    if (input.ndim() != 3)
        throw std::runtime_error("3-channel image must be 3 dims ");

    py::buffer_info buf = input.request();

    cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char *)buf.ptr);

    return mat;
}
py::array_t<unsigned char> cv_mat_uint8_1c_to_numpy(cv::Mat &input)
{

    py::array_t<unsigned char> dst = py::array_t<unsigned char>({input.rows, input.cols}, input.data);
    return dst;
}

py::array_t<unsigned char> sgm_c_vpi(py::array_t<unsigned char> &left_image, py::array_t<unsigned char> &right_image)
{
    cv::Mat left_image_mat = numpy_uint8_3c_to_cv_mat(left_image_np);
    cv::Mat right_image_mat = numpy_uint8_3c_to_cv_mat(right_image_np);
    cv::Mat left_image_mat_g, right_image_mat_g;
    cv::cvtColor(left_image_mat, left_image_mat_g, cv::COLOR_RGB2GRAY);
    cv::cvtColor(right_image_mat, right_image_mat_g, cv::COLOR_RGB2GRAY);

    // SGM block
    cv::Mat disparity;
    disparity = sgm(left_image_mat_g, right_image_mat_g);
    // SGM block end

    return cv_mat_uint8_1c_to_numpy(disparity);
}

PYBIND11_MODULE(sgm, m)
{
    m.doc() = "c++ sgm VPI";

    m.def("sgm_c_vpi", &left_image, &right_image);
    m.def("camera_calibration")
}
// Pybind11 block end


void camera_calibration()
{
    cv::Mat cam_left(3, 3, cv::DataType<float>::type);
    cam_left.at<float>(0, 0) = 2937.96522281f;
    cam_left.at<float>(0, 1) = 0.0f;
    cam_left.at<float>(0, 2) = 1803.97085512f;

    cam_left.at<float>(1, 0) = 0.0f;
    cam_left.at<float>(1, 1) = 3943.93428919f;
    cam_left.at<float>(1, 2) = 1272.78948524f;

    cam_left.at<float>(2, 0) = 0.0f;
    cam_left.at<float>(2, 1) = 0.0f;
    cam_left.at<float>(2, 2) = 1.0f;

    cv::Mat dist_left(5, 1, cv::DataType<float>::type);
    dist_left.at<float>(0, 0) = -0.0563045653025f;
    dist_left.at<float>(1, 0) = 0.473018514728f;
    dist_left.at<float>(2, 0) = 0.000419477763134f;
    dist_left.at<float>(3, 0) = 0.00120517316419f;
    dist_left.at<float>(4, 0) = -1.05028409189f;

    cv::Mat cam_right(3, 3, cv::DataType<float>::type);
    cam_right.at<float>(0, 0) = 2991.94118199f;
    cam_right.at<float>(0, 1) = 0.0f;
    cam_right.at<float>(0, 2) = 1810.39424309f;

    cam_right.at<float>(1, 0) = 0.0f;
    cam_right.at<float>(1, 1) = 4011.3879452f;
    cam_right.at<float>(1, 2) = 1124.31708597f;

    cam_right.at<float>(2, 0) = 0.0f;
    cam_right.at<float>(2, 1) = 0.0f;
    cam_right.at<float>(2, 2) = 1.0f;

    cv::Mat dist_right(5, 1, cv::DataType<float>::type);
    dist_right.at<float>(0, 0) = -0.178320508006f;
    dist_right.at<float>(1, 0) = 0.958988270116f;
    dist_right.at<float>(2, 0) = -0.00223758101124f;
    dist_right.at<float>(3, 0) = -0.000434069367022f;
    dist_right.at<float>(4, 0) = -1.40516826033f;

    IplImage *Left_Mapx = cvCreateImage(cvSize(1280, 960), IPL_DEPTH_32F, 1);
    IplImage *Left_Mapy = cvCreateImage(cvSize(1280, 960), IPL_DEPTH_32F, 1);
    IplImage *Right_Mapx = cvCreateImage(cvSize(1280, 960), IPL_DEPTH_32F, 1);
    IplImage *Right_Mapy = cvCreateImage(cvSize(1280, 960), IPL_DEPTH_32F, 1);
    CvMat *Rl = cvCreateMat(3, 3, CV_64F);
    CvMat *Rr = cvCreateMat(3, 3, CV_64F);
    CvMat *Pl = cvCreateMat(3, 4, CV_64F);
    CvMat *Pr = cvCreateMat(3, 4, CV_64F);

    cvStereoRectify(Intrinsics_Camera_Left, Intrinsics_Camera_Right,
                    Distortion_Camera_Left, Distortion_Camera_Right,
                    cvSize(1280, 960), R_opencv, Translation_matlab,
                    Rl, Rr, Pl, Pr, 0, 2048, 0);

    cvInitUndistortRectifyMap(Intrinsics_Camera_Left, Distortion_Camera_Left, Rl, Pl,
                              Left_Mapx, Left_Mapy);
    cvInitUndistortRectifyMap(Intrinsics_Camera_Right, Distortion_Camera_Right, Rr, Pr,
                              Right_Mapx, Right_Mapy);

    cvSaveImage("Left_Mapx.png", Left_Mapx);
    cvSaveImage("Left_Mapy.png", Left_Mapy);
    cvSaveImage("Right_Mapx.png", Right_Mapx);
    cvSaveImage("Right_Mapy.png", Right_Mapy);
}

cv::Mat sgm(cv::Mat left_image, cv::Mat right_image)
{
    #define CHECK_STATUS(STMT)                                    \
    do                                                        \
    {                                                         \
        VPIStatus status = (STMT);                            \
        if (status != VPI_SUCCESS)                            \
        {                                                     \
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
            vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
            std::ostringstream ss;                            \
            ss << vpiStatusGetName(status) << ": " << buffer; \
            throw std::runtime_error(ss.str());               \
        }                                                     \
    } while (0);
    
    //read calibration matrix
    IplImage *Left_Mapx = cvLoadImage("Left_Mapx.png");
    IplImage *Left_Mapy = cvLoadImage("Left_Mapy.png");
    IplImage *Right_Mapx = cvLoadImage("Right_Mapx.png");
    IplImage *Right_Mapy = cvLoadImage("Right_Mapy.png");
    //read calibration matrix end

    cv::namedWindow("disparity", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("CSI Camera left", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("CSI Camera right", cv::WINDOW_AUTOSIZE);
    VPIImage left = NULL;
    VPIImage right = NULL;
    VPIImage disparity = NULL;
    VPIStream stream = NULL;
    VPIPayload stereo = NULL;
    cv::resize(cvImageLeft, cvImageLeft, cv::Size(480, 270));
    cv::resize(cvImageRight, cvImageRight, cv::Size(480, 270));

    cv::imshow("CSI Camera left", cvImageLeft);
    cv::imshow("CSI Camera right", cvImageRight);
    cv::waitKey(5000);
    cvRemap(IplimgLeft, img_left_Change, Left_Mapx, Left_Mapy);
    cvRemap(IplimgRight, img_right_Change, Right_Mapx, Right_Mapy);

    //  Nvidia SGM VPI
    try
    {
        cvImageLeft.convertTo(cvImageLeft, CV_16UC1);
        cvImageRight.convertTo(cvImageRight, CV_16UC1);

        // Now parse the backend
        VPIBackend backendType;

        // backendType = VPI_BACKEND_CPU;
        backendType = VPI_BACKEND_CUDA;
        // backendType = VPI_BACKEND_PVA;

        // Create the stream for the given backend.
        CHECK_STATUS(vpiStreamCreate(backendType, &stream));

        // We now wrap the loaded images into a VPIImage object to be used by VPI.
        {
            // First fill VPIImageData with the, well, image data...
            VPIImageData imgData;
            memset(&imgData, 0, sizeof(imgData));
            imgData.type = VPI_IMAGE_FORMAT_U16;
            imgData.numPlanes = 1;
            imgData.planes[0].width = cvImageLeft.cols;
            imgData.planes[0].height = cvImageLeft.rows;
            imgData.planes[0].pitchBytes = cvImageLeft.step[0];
            imgData.planes[0].data = cvImageLeft.data;

            // Wrap it into a VPIImage. VPI won't make a copy of it, so the original
            // image must be in scope at all times.
            CHECK_STATUS(vpiImageCreateHostMemWrapper(&imgData, 0, &left));

            imgData.planes[0].width = cvImageRight.cols;
            imgData.planes[0].height = cvImageRight.rows;
            imgData.planes[0].pitchBytes = cvImageRight.step[0];
            imgData.planes[0].data = cvImageRight.data;

            CHECK_STATUS(vpiImageCreateHostMemWrapper(&imgData, 0, &right));
        }

        // Create the image where the disparity map will be stored.
        CHECK_STATUS(vpiImageCreate(cvImageLeft.cols, cvImageLeft.rows, VPI_IMAGE_FORMAT_U16, 0, &disparity));

        // Create the payload for Harris Corners Detector algorithm

        VPIStereoDisparityEstimatorParams params;
        params.windowSize = 5;
        params.maxDisparity = 64;
        CHECK_STATUS(vpiCreateStereoDisparityEstimator(backendType, cvImageLeft.cols, cvImageLeft.rows,
                                                       VPI_IMAGE_FORMAT_U16, params.maxDisparity, &stereo));

        // Submit it with the input and output images
        CHECK_STATUS(vpiSubmitStereoDisparityEstimator(stream, stereo, left, right, disparity, &params));

        // Wait until the algorithm finishes processing
        CHECK_STATUS(vpiStreamSync(stream));

        // Now let's retrieve the output
        {
            // Lock output to retrieve its data on cpu memory
            VPIImageData data;
            CHECK_STATUS(vpiImageLock(disparity, VPI_LOCK_READ, &data));

            // Make an OpenCV matrix out of this image
            cv::Mat cvOut(data.planes[0].height, data.planes[0].width, CV_16UC1, data.planes[0].data,
                          data.planes[0].pitchBytes);

            // Scale result and write it to disk
            double min, max;
            minMaxLoc(cvOut, &min, &max);
            cvOut.convertTo(cvOut, CV_8UC1, 255.0 / (max - min), -min);

            // std::cout << "success save"
            //           << "\n";
            // imwrite("disparity.png", cvOut);
            cv::imshow("disparity", cvOut);
            cv::waitKey(30);

            // Done handling output, don't forget to unlock it.
            CHECK_STATUS(vpiImageUnlock(disparity));
        }
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        retval = 1;
    }
    //  Nvidia SGM VPI end

    int main(int argc, char *argv[])
    {
        if (stream != NULL)
        {
            vpiStreamSync(stream);
        }

        vpiImageDestroy(left);
        vpiImageDestroy(right);
        vpiImageDestroy(disparity);
        vpiPayloadDestroy(stereo);
        vpiStreamDestroy(stream);

        return retval;
    }
