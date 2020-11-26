#include <opencv2/core/version.hpp>
#if CV_MAJOR_VERSION >= 3
#include <opencv2/imgcodecs.hpp>
#else
#include <opencv2/highgui/highgui.hpp>
#endif

#include <opencv2/imgproc/imgproc.hpp>

#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/algo/StereoDisparityEstimator.h>
#include <opencv2/opencv.hpp>
#include <cstring> // for memset
#include <iostream>

#define CHECK_STATUS(STMT)                                      \
    do                                                          \
    {                                                           \
        VPIStatus status = (STMT);                              \
        if (status != VPI_SUCCESS)                              \
        {                                                       \
            throw std::runtime_error(vpiStatusGetName(status)); \
        }                                                       \
    } while (0);

std::string gstreamer_pipeline(int sensor_id, int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method)
{
    return "nvarguscamerasrc sensor-id=" + std::to_string(sensor_id) + " ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

// int main()
// {
//     int sensor_id;
//     int capture_width = 1280;
//     int capture_height = 720;
//     int display_width = 1280;
//     int display_height = 720;
//     int framerate = 60;
//     int flip_method = 2;

//     sensor_id = 0;
//     std::string pipeline = gstreamer_pipeline(sensor_id, capture_width,
//                                               capture_height,
//                                               display_width,
//                                               display_height,
//                                               framerate,
//                                               flip_method);
//     sensor_id = 1;
//     std::string pipeline_second = gstreamer_pipeline(sensor_id, capture_width,
//                                                      capture_height,
//                                                      display_width,
//                                                      display_height,
//                                                      framerate,
//                                                      flip_method);

//     std::cout << "Using pipeline: \n\t" << pipeline << "  and  " << pipeline_second << "\n";

//     cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
//     cv::VideoCapture cap_second(pipeline_second, cv::CAP_GSTREAMER);

//     if (!cap.isOpened() || !cap_second.isOpened())
//     {
//         std::cout << "Failed to open camera." << std::endl;
//         return (-1);
//     }

//     cv::namedWindow("CSI Camera", cv::WINDOW_AUTOSIZE);
//     cv::namedWindow("CSI Camera 2", cv::WINDOW_AUTOSIZE);

//     cv::Mat img, img_second;

//     std::cout << "Hit ESC to exit"
//               << "\n";
//     while (true)
//     {
//         if (!cap.read(img) || !cap_second.read(img_second))
//         {
//             std::cout << "Capture read error" << std::endl;
//             break;
//         }

//         cv::imshow("CSI Camera", img);
//         cv::imshow("CSI Camera 2", img_second);

//         int keycode = cv::waitKey(30) & 0xff;
//         if (keycode == 27)
//             break;
//     }

//     cap.release();
//     cap_second.release();
//     cv::destroyAllWindows();
//     return 0;
// }

int main(int argc, char *argv[])
{
    // Camera read
    int sensor_id;
    int capture_width = 1280;
    int capture_height = 720;
    int display_width = 1280;
    int display_height = 720;
    int framerate = 60;
    int flip_method = 2;

    std::string pipeline = gstreamer_pipeline(0, capture_width,
                                              capture_height,
                                              display_width,
                                              display_height,
                                              framerate,
                                              flip_method);

    std::string pipeline_second = gstreamer_pipeline(1, capture_width,
                                                     capture_height,
                                                     display_width,
                                                     display_height,
                                                     framerate,
                                                     flip_method);

    std::cout << "Using pipeline: \n\t" << pipeline << "  and  " << pipeline_second << "\n";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    cv::VideoCapture cap_second(pipeline_second, cv::CAP_GSTREAMER);

    if (!cap.isOpened() || !cap_second.isOpened())
    {
        std::cout << "Failed to open camera." << std::endl;
        return (-1);
    }

    cv::namedWindow("CSI Camera", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("CSI Camera 2", cv::WINDOW_AUTOSIZE);

    cv::Mat img, img_second;

    VPIImage left = NULL;
    VPIImage right = NULL;
    VPIImage disparity = NULL;
    VPIStream stream = NULL;
    VPIPayload stereo = NULL;

    int retval = 0;

    try
    {
        // Load the input images
        cv::Mat cvImageLeft, cvImageRight;

        if (!cap.read(cvImageLeft) || !cap_second.read(cvImageRight))
        {
            std::cout << "Capture read error" << std::endl;
        }

        cv::cvtColor(cvImageLeft, cvImageLeft, cv::COLOR_BGR2GRAY);
        cv::cvtColor(cvImageRight, cvImageRight, cv::COLOR_BGR2GRAY);
        cv::resize(cvImageLeft, cvImageLeft,cv::Size(480,270));
        cv::resize(cvImageRight, cvImageRight,cv::Size(480,270));

        //// For image
        // cv::Mat cvImageLeft = cv::imread(strLeftFileName, cv::IMREAD_GRAYSCALE);
        // if (cvImageLeft.empty())
        // {
        //     throw std::runtime_error("Can't open '" + strLeftFileName + "'");
        // }
        // cv::Mat cvImageRight = cv::imread(strRightFileName, cv::IMREAD_GRAYSCALE);
        // if (cvImageRight.empty())
        // {
        //     throw std::runtime_error("Can't open '" + strRightFileName + "'");
        // }

        // Currently we only accept unsigned 16bpp inputs.
        cvImageLeft.convertTo(cvImageLeft, CV_16UC1);
        cvImageRight.convertTo(cvImageRight, CV_16UC1);

        // Now process the device type
        VPIDeviceType devType;

        // devType = VPI_DEVICE_TYPE_CUDA;
        devType = VPI_BACKEND_CPU;

        // Create the stream for the given backend.
        CHECK_STATUS(vpiStreamCreate(devType, &stream));

        // We now wrap the loaded images into a VPIImage object to be used by VPI.
        {
            // First fill VPIImageData with the, well, image data...
            VPIImageData imgData;
            memset(&imgData, 0, sizeof(imgData));
            imgData.type = VPI_IMAGE_TYPE_U16;
            imgData.numPlanes = 1;
            imgData.planes[0].width = cvImageLeft.cols;
            imgData.planes[0].height = cvImageLeft.rows;
            imgData.planes[0].rowStride = cvImageLeft.step[0];
            imgData.planes[0].data = cvImageLeft.data;

            // Wrap it into a VPIImage. VPI won't make a copy of it, so the original
            // image must be in scope at all times.
            CHECK_STATUS(vpiImageWrapHostMem(&imgData, 0, &left));

            imgData.planes[0].width = cvImageRight.cols;
            imgData.planes[0].height = cvImageRight.rows;
            imgData.planes[0].rowStride = cvImageRight.step[0];
            imgData.planes[0].data = cvImageRight.data;

            CHECK_STATUS(vpiImageWrapHostMem(&imgData, 0, &right));
        }

        // Create the image where the disparity map will be stored.
        CHECK_STATUS(vpiImageCreate(cvImageLeft.cols, cvImageLeft.rows, VPI_IMAGE_TYPE_U16, 0, &disparity));

        // Create the payload for Harris Corners Detector algorithm

        VPIStereoDisparityEstimatorParams params;
        params.windowSize = 5;
        params.maxDisparity = 128;
        CHECK_STATUS(vpiCreateStereoDisparityEstimator(stream, cvImageLeft.cols, cvImageLeft.rows, VPI_IMAGE_TYPE_U16,
                                                       params.maxDisparity, &stereo));

        // Submit it with the input and output images
        CHECK_STATUS(vpiSubmitStereoDisparityEstimator(stereo, left, right, disparity, &params));

        // Wait until the algorithm finishes processing
        CHECK_STATUS(vpiStreamSync(stream));

        // Now let's retrieve the output
        {
            // Lock output to retrieve its data on cpu memory
            VPIImageData data;
            CHECK_STATUS(vpiImageLock(disparity, VPI_LOCK_READ, &data));

            // Make an OpenCV matrix out of this image
            cv::Mat cvOut(data.planes[0].height, data.planes[0].width, CV_16UC1, data.planes[0].data,
                          data.planes[0].rowStride);

            // Scale result and write it to disk
            double min, max;
            minMaxLoc(cvOut, &min, &max);
            cvOut.convertTo(cvOut, CV_8UC1, 255.0 / (max - min), -min);

            std::cout << "disparity success" << "\n";
            imwrite("disparity.png", cvOut);

            // Done handling output, don't forget to unlock it.
            CHECK_STATUS(vpiImageUnlock(disparity));
        }
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        retval = 1;
    }

    // Clean up

    // Make sure stream is synchronized before destroying the objects
    // that might still be in use.
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
