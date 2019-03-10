#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

print(tf.VERSION)

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
plt.interactive(False)

from pathlib import *

data_root = Path('/home/charles/Downloads/dataset5/A')

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory(str(data_root))


# In[3]:


classifier_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2" #@param {type:"string"}

import tensorflow_hub as hub
from tensorflow.keras import layers


def classifier(x):
  classifier_module = hub.Module(classifier_url)
  return classifier_module(x)
  
IMAGE_SIZE = hub.get_expected_image_size(hub.Module(classifier_url))

classifier_layer = layers.Lambda(classifier, input_shape = IMAGE_SIZE+[3])
classifier_model = tf.keras.Sequential([classifier_layer])
classifier_model.summary()

image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SIZE)
for image_batch,label_batch in image_data:
  print("Image batch shape: ", image_batch.shape)
  print("Label batch shape: ", label_batch.shape)
  break


# In[4]:


import tensorflow.keras.backend as K
sess = K.get_session()
init = tf.global_variables_initializer()

sess.run(init)


# In[5]:


feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2" #@param {type:"string"}

def feature_extractor(x):
  feature_extractor_module = hub.Module(feature_extractor_url)
  return feature_extractor_module(x)

IMAGE_SIZE = hub.get_expected_image_size(hub.Module(feature_extractor_url))

features_extractor_layer = layers.Lambda(feature_extractor, input_shape=IMAGE_SIZE+[3])
features_extractor_layer.trainable = False


# In[7]:


model = tf.keras.Sequential([
  features_extractor_layer,
  layers.Dense(image_data.num_classes, activation='softmax')
])
model.summary()


# In[8]:


init = tf.global_variables_initializer()
sess.run(init)
result = model.predict(image_batch)
result.shape


# In[14]:


from tensorflow.keras import optimizers
model.compile(
  optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
  loss='categorical_crossentropy',
  metrics=['accuracy'])

class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []
    
  def on_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])

steps_per_epoch = image_data.samples//image_data.batch_size
batch_stats = CollectBatchStats()
model.fit((item for item in image_data), epochs=1, 
                    steps_per_epoch=steps_per_epoch,
                    callbacks = [batch_stats])


# In[15]:


import matplotlib.pylab as plt
plt.style.use('dark_background')

plt.figure()
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(batch_stats.batch_losses)

plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(batch_stats.batch_acc)


# In[16]:


label_names = sorted(image_data.class_indices.items(), key=lambda pair:pair[1])
label_names = np.array([key.title() for key, value in label_names])
label_names


# In[17]:


result_batch = model.predict(image_batch)

labels_batch = label_names[np.argmax(result_batch, axis=-1)]


plt.figure(figsize=(10,9))
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  plt.title(labels_batch[n])
  plt.axis('off')
_ = plt.suptitle("Model predictions")

model.save('sign_model.h5')


# In[4]:


import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

cap = cv2.VideoCapture(0)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

print('initialized')

font = cv2.FONT_HERSHEY_SIMPLEX

model = load_model('sign_model.h5')

while(True):
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
        #stdev =
        if (area > max_area and True):
            max_area = area
            ci = i

        # Largest area contour
    cnts = contours[ci]

    border = 50
    x, y, w, h = cv2.boundingRect(cnts)
    cv2.rectangle(frame, (x-border, y-border), (x + w + border, y + h + border), (0, 255, 0), 2)

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


# In[30]:





# In[ ]:




