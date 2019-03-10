import numpy as np
import tensorflow as tf
import cv2

cap = cv2.VideoCapture(0)


export_dir='./saved_models/1550962723/saved_model.pb'
with tf.Session(graph=tf.Graph()) as sess:
  tf.saved_model.loader.load(sess, [tag_constants.TRAINING], export_dir)




while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    lower_skin = np.array([0, 133, 77])
    upper_skin = np.array([255, 173, 127])

    mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

    # Kernel matrices for morphological transformation
    kernel_square = np.ones((11, 11), np.uint8)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Perform morphological transformations to filter out the background noise
    # Dilation increase skin color area
    # Erosion increase skin color area
    dilation = cv2.dilate(mask, kernel_ellipse, iterations=1)
    erosion = cv2.erode(dilation, kernel_square, iterations=1)
    dilation2 = cv2.dilate(erosion, kernel_ellipse, iterations=1)
    filtered = cv2.medianBlur(dilation2, 5)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    dilation2 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilation3 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
    median = cv2.medianBlur(dilation2, 5)
    ret, thresh = cv2.threshold(median, 127, 255, 0)


    # Find contours of the filtered frame
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw Contours
    # cv2.drawContours(frame, cnt, -1, (122,122,0), 3)
    # cv2.imshow('Dilation',median)

    # Find Max contour area (Assume that hand is in the frame)
    max_area = 100
    ci = 0
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        #stdev =
        if (area > max_area and True):
            max_area = area
            ci = i

        # Largest area contour
    cnts = contours[ci]

    # Find convex hull
    hull = cv2.convexHull(cnts)

    # Find convex defects
    hull2 = cv2.convexHull(cnts, returnPoints=False)
    defects = cv2.convexityDefects(cnts, hull2)

    # Get defect points and draw them in the original image
    FarDefect = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnts[s][0])
        end = tuple(cnts[e][0])
        far = tuple(cnts[f][0])
        FarDefect.append(far)
        cv2.line(frame, start, end, [0, 255, 0], 1)
        cv2.circle(frame, far, 10, [100, 255, 255], 3)

    # Find moments of the largest contour
    moments = cv2.moments(cnts)

    # Central mass of first order moments
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
        cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
    centerMass = (cx, cy)

    border = 50
    x, y, w, h = cv2.boundingRect(cnts)
    cv2.rectangle(frame, (x-border, y-border), (x + w + border, y + h + border), (0, 255, 0), 2)

    cv2.imshow("test", frame)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

