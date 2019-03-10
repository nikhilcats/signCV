import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

cap = cv2.VideoCapture(0)
print('camera')
sess = tf.Session()
sess.run(tf.initialize_all_variables())

print('initialized')

font = cv2.FONT_HERSHEY_SIMPLEX

feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2" #@param {type:"string"}

model = load_model('sign_model.h5')

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        breaktf.Session

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
        # stdev =
        if (area > max_area and True):
            max_area = area
            ci = i

        # Largest area contour
    cnts = contours[ci]

    border = 50
    x, y, w, h = cv2.boundingRect(cnts)
    cv2.rectangle(frame, (x - border, y - border), (x + w + border, y + h + border), (0, 255, 0), 2)

    crop_img = frame[y - border:y + h + border, x - border:x + w + border]

    img2 = cv2.resize(crop_img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    np_image_data = np.asarray(img2).astype('float32') / 255
    np_image = cv2.normalize(np_image_data.astype('float'), None, 0, 1, cv2.NORM_MINMAX)
    np_final = np.expand_dims(np_image_data, axis=0)
    print(np_final.shape)
    result = model.predict(np_final)
    print(result)
    label = label_names[np.argmax(result, axis=-1)]
    cv2.putText(frame, label, (cx, cy), font, 4, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("test", frame)
    break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()