########################################
# Caffe Model, Face Tracking... Working...
# 1. area check -> forward/ backward
# 2. yaw
#****************************************

# import the necessary packages
import numpy as np
import argparse
import cv2
import imutils
from imutils.video import VideoStream
from djitellopy import tello

#defining argument parsers
# ap  = argparse.ArgumentParser()
# ap.add_argument("-i","--image",required=True,help="Input image path")
# args = vars(ap.parse_args())

# tello init operation
##send_rc_control(left_right, forward_backward, up_down, yaw)
me = tello.Tello()
# me.connect()
# print(me.get_battery())
# me.streamon()
# me.takeoff()
# me.send_rc_control(0, 0, 15, 0)

# webcam init
# cap = cv2.VideoCapture(0)

# # 비디오 처리
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('./video_save/output.avi', fourcc, 25.0, (800, 450))

#defining prototext and caffemodel paths
caffeModel = "./models/res10_300x300_ssd_iter_140000.caffemodel"
prototextPath = "./models/deploy.prototxt.txt"

#Load Model
print("Loading model...................")
net = cv2.dnn.readNetFromCaffe(prototextPath,caffeModel)

vs = VideoStream(src=0).start()

# def adjust_tello_position(offset_x, offset_y, offset_z):
def adjust_tello_position(offset_x,  offset_z):
    """
    Adjusts the position of the tello drone based on the offset values given from the frame

    :param offset_x: Offset between center and face x coordinates      (yaw)
    :param offset_y: Offset between center and face y coordinates      (ud)
    :param offset_z: Area of the face detection rectangle on the frame (fb)
    """
    # send_rc_control(left_right, forward_backward, up_down, yaw)
    if not -90 <= offset_x <= 90 and offset_x is not 0:
        if offset_x < 0:
            me.send_rc_control(0, 0, 0, -15)
            # me.rotate_ccw(10)
        elif offset_x > 0:
            me.send_rc_control(0, 0, 0, 15)
            # me.rotate_cw(10)

    # if not -70 <= offset_y <= 70 and offset_y is not -30:
    #     if offset_y < 0:
    #         drone.move_up(20)
    #     elif offset_y > 0:
    #         drone.move_down(20)

    if not 40000 <= offset_z <= 50000 and offset_z is not 0:
        if offset_z < 40000:
            # me.move_forward(20)
            me.send_rc_control(0, 20, 0, 0)
        elif offset_z > 50000:
            # me.move_back(20)
            me.send_rc_control(0, -20, 0, 0)

# tello cam init
# frame_read = me.get_frame_read()

while True:
    image = vs.read()
    # image = frame_read.frame
    image = imutils.resize(image, width=800)

    # extract the dimensions , Resize image into 300x300 and converting image into blobFromImage
    (h,w) = image.shape[:2]
    print('h= ', h, 'w= ', w)

    # # 비디오 처리
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # out = cv2.VideoWriter('./video_save/output.avi', fourcc, 25.0, (w, h))

    # blobImage convert RGB (104.0, 177.0, 123.0)
    blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),1.0,(300,300),(104.0, 177.0, 123.0))

    #passing blob through the network to detect and pridiction
    net.setInput(blob)
    detections = net.forward()

    # Calculate frame center
    center_x = int(w / 2)
    center_y = int(h / 2)

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence and prediction

        confidence = detections[0, 0, i, 2]

        # filter detections by confidence greater than the minimum confidence
        # print(confidence)
        if confidence > 0.5:
            print(confidence)
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            # y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY + 8), (endX, endY),
                          (0 , 255, 0), 2)
            cv2.putText(image, text, (startX, startY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            # compute face area
            cx = (startX + endX) // 2
            cy = (startY + endY) // 2
            zarea = (endX - startX) * (endY - startY)
            print('face size : ', zarea)

            # compute offset from center
            offset_x = cx - center_x
            offset_y = cy - center_y
            print('offset_x', offset_x)

            cv2.putText(image, "area: "+ str(zarea), (5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(image, "offset x: " + str(offset_x), (5, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            #center of face
            cv2.circle(image, (cx, cy), 8, (0, 0, 255), 3)

            # center of frame
            cv2.circle(image, (center_x, center_y), 8, (255, 0, 255), 3)

            # drone control
            # adjust_tello_position(offset_x, offset_y, z_area)
            adjust_tello_position(offset_x, zarea)

    # 비디오 저장
    out.write(image)

    # show the output image
    cv2.imshow("Output", image)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    elif key == ord("l"):
        me.land()
    elif key == ord("r"):
        me.takeoff()
        me.send_rc_control(0, 0, 15, 0)

# do a bit of cleanup
out.write(image)
me.land()
# frame_read.stopped
cv2.destroyAllWindows()
vs.stop()