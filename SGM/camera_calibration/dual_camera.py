# MIT License
# Copyright (c) 2019,2020 JetsonHacks
# See license
# A very simple code snippet
# Using two  CSI cameras (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit (Rev B01) using OpenCV
# Drivers for the camera and OpenCV are included in the base image in JetPack 4.3+

# This script will open a window and place the camera stream from each camera in a window
# arranged horizontally.
# The camera streams are each read in their own thread, as when done sequentially there
# is a noticeable lag
# For better performance, the next step would be to experiment with having the window display
# in a separate thread

import cv2
import threading
import numpy as np

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of each camera pane in the window on the screen

left_camera = None
right_camera = None
CAMERA_WH=(3264 ,2464)


class CSI_Camera:

    def __init__(self):
        # Initialize instance variables
        # OpenCV video capture element
        self.video_capture = None
        # The last captured image from the camera
        self.frame = None
        self.grabbed = False
        # The thread where the video capture runs
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False

    def open(self, gstreamer_pipeline_string):
        try:
            self.video_capture = cv2.VideoCapture(
                gstreamer_pipeline_string, cv2.CAP_GSTREAMER
            )

        except RuntimeError:
            self.video_capture = None
            print("Unable to open camera")
            print("Pipeline: " + gstreamer_pipeline_string)
            return
        # Grab the first frame to start the video capturing
        self.grabbed, self.frame = self.video_capture.read()

    def start(self):
        if self.running:
            print('Video capturing is already running')
            return None
        # create a thread to read the camera image
        if self.video_capture != None:
            self.running = True
            self.read_thread = threading.Thread(target=self.updateCamera)
            self.read_thread.start()
        return self

    def stop(self):
        self.running = False
        self.read_thread.join()

    def updateCamera(self):
        # This is the thread to read images from the camera
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            except RuntimeError:
                print("Could not read image from camera")
        # FIX ME - stop and cleanup thread
        # Something bad happened

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def release(self):
        if self.video_capture != None:
            self.video_capture.release()
            self.video_capture = None
        # Now kill the thread
        if self.read_thread != None:
            self.read_thread.join()


# Currently there are setting frame rate on CSI Camera on Nano through gstreamer
# Here we directly select sensor_mode 3 (1280x720, 59.9999 fps)
def gstreamer_pipeline(
    sensor_id=0,
    sensor_mode=3,
    capture_width=CAMERA_WH[0],
    capture_height=CAMERA_WH[1],
    display_width=CAMERA_WH[0],
    display_height=CAMERA_WH[1],
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d sensor-mode=%d ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            sensor_mode,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def start_cameras():
    left_camera = CSI_Camera()
    left_camera.open(
        gstreamer_pipeline(
            sensor_id=0,
            sensor_mode=3,
            flip_method=2,
            display_height=CAMERA_WH[1],
            display_width=CAMERA_WH[0]
        )
    )
    left_camera.start()

    right_camera = CSI_Camera()
    right_camera.open(
        gstreamer_pipeline(
            sensor_id=1,
            sensor_mode=3,
            flip_method=2,
            display_height=CAMERA_WH[1],
            display_width=CAMERA_WH[0]
        )
    )
    right_camera.start()

    cv2.namedWindow("CSI Cameras", cv2.WINDOW_AUTOSIZE)
    # cv2.namedWindow("CSI Cameras left", cv2.WINDOW_AUTOSIZE)
    # cv2.namedWindow("CSI Cameras right", cv2.WINDOW_AUTOSIZE)

    if (
        not left_camera.video_capture.isOpened()
        or not right_camera.video_capture.isOpened()
    ):
        # Cameras did not open, or no camera attached

        print("Unable to open any cameras")
        # TODO: Proper Cleanup
        SystemExit(0)

    saveimg_Count = 0

    while cv2.getWindowProperty("CSI Cameras", 0) >= 0:

        _, left_image = left_camera.read()
        _, right_image = right_camera.read()

        # left_image_copy = left_image.copy()
        # right_image_copy = right_image.copy()
        # left_image_copy = cv2.resize(left_image_copy, CAMERA_WH)
        # # right_image_copy = cv2.resize(right_image_copy, CAMERA_WH)

        cam_left = np.array([3014.02900, 0.0,  4039.35971,
                             0.0, 1498.17668 , 1069.70864,
                             0.0, 0.0, 1.0])
        cam_left= cam_left.reshape(3,3)

        dist_left = np.array([-0.07729  , 0.19770  , -0.00630 ,  0.01405 , 0.00000])
                              
        cam_right = np.array([3080.02213, 0.0, 4128.45382, 
                              0.0, 1554.16884, 1346.32313,
                              0.0, 0.0, 1.0])
        cam_right= cam_right.reshape(3,3)

        dist_right = np.array([-0.14714 ,  0.41824  , 0.00563,   0.00896,  0.00000])
                                
        om = np.array([0.05067, -0.01627, 0.00248])
        R = cv2.Rodrigues(om)[0]
        T = np.array([-59.03521, 1.19002, -0.89700])

        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cam_left, dist_left,
                                                                          cam_right, dist_right, CAMERA_WH, R,
                                                                          T)

        left_map1, left_map2 = cv2.initUndistortRectifyMap(cam_left, dist_left,R1, P1, CAMERA_WH, cv2.CV_16SC2)
        right_map1, right_map2 = cv2.initUndistortRectifyMap(cam_right, dist_right, R2, P2, CAMERA_WH, cv2.CV_16SC2)

        img1_rectified = cv2.remap(left_image, left_map1, left_map2, cv2.INTER_LINEAR)
        img2_rectified = cv2.remap(right_image, right_map1, right_map2, cv2.INTER_LINEAR)

        # cv2.initUndistortRectifyMap(cam_left, dist_left)
        img1_rectified = cv2.resize(img1_rectified,(954,720))
        img2_rectified = cv2.resize(img2_rectified,(954,720))

        camera_images = np.hstack((img1_rectified, img2_rectified))

        # cv2.imshow("CSI Cameras left", img1_rectified)
        # cv2.imshow("CSI Cameras right", img2_rectified)

        cv2.imshow("CSI Cameras", camera_images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('left'+str(saveimg_Count)+'.jpg', left_image)
            cv2.imwrite('right'+str(saveimg_Count)+'.jpg', right_image)
            saveimg_Count += 1
            print('success')

    left_camera.stop()
    left_camera.release()
    right_camera.stop()
    right_camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_cameras()
