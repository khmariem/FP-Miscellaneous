import argparse
import numpy as np
import scipy.misc
import cv2 as cv
import pyrealsense2 as rs

def capture(pattern_size, square_size, disp):

    nb = 0
    images = []
    # Configure color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        while True:

            # Wait for a coherent pair of color frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # Show images
            cv.namedWindow('RealSense', cv.WINDOW_AUTOSIZE)
            cv.imshow('RealSense', color_image)
            key = cv.waitKey(10)

            if key==ord("c") and nb<10:
                nb+=1
                print('Saving image '+str(nb))
                images.append(str(nb)+'.jpg')
                scipy.misc.imsave(str(nb)+'.jpg', color_image)
                continue
            elif nb==10:
                cv.destroyAllWindows()
                return intrinsic_matrix(images, pattern_size, square_size, disp)
            else:
                continue

    finally:
        # Stop streaming
        pipeline.stop()




def intrinsic_matrix(images, pattern_size, square_size, disp=False):

    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1,2)
    pattern_points*=square_size

    image_points = []
    object_points = []
    B=[]

    for img in images:
        #load image as grayscale
        image = cv.imread(img)
        b, chess_corners = cv.findChessboardCorners(image, pattern_size, None)
        print(chess_corners)

        #append extracted information for every image
        image_points.append(chess_corners)
        object_points.append(pattern_points)
        B.append(b)

    if True in B:
        img = cv.imread(images[0])
        height, width = img.shape[:2]
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(object_points, image_points, (width, height), None, None)

        #MAYBE ADD SUBPIXEL REFINEMENT: TODO

        if disp:
            pic = cv.drawChessboardCorners(img, pattern_size, image_points[0],B[0])
            cv.namedWindow('RealSense', cv.WINDOW_AUTOSIZE)
            cv.imshow("detected corners", pic)
            cv.waitKey(10000)
            cv.destroyAllWindows()
        
        np.save('calibration', cameraMatrix)
    else:
        cameraMatrix = None

    return cameraMatrix

if __name__=="__main__":
    #for displaying first picture
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--disp',type=bool, help='Display the found corners on the chess board')
    parser.add_argument('-p','--pattern',type=int, help='The correct pattern size. Please choose carefully, otherwise it will not work')
    args = parser.parse_args()

    pattern = args.pattern
    pattern = (pattern, pattern)
    square_size = 1.0
    disp = args.disp

    cameraMatrix = capture(pattern, square_size, disp)
    print(cameraMatrix)