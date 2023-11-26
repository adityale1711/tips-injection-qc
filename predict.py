import os
import cv2
import math
import pickle
import config
import imutils
import numpy as np
from tkinter import Tk, filedialog
from keras.models import load_model
from keras.utils import load_img, img_to_array

def calculate_concentricity(innerX, innerY, innerRadius, outerX, outerY, outerRadius):
    distance_between_centers = math.sqrt((innerX - outerX)**2 + (innerY - outerY)**2)
    total_radius = innerRadius + outerRadius

    concentricity = 1 - (distance_between_centers / total_radius)
    return concentricity

# Determine input file and Open image directory dialog
Tk().withdraw()
imagePaths = filedialog.askopenfilename(title='Select Image Directory')
# imagePaths = 'input_test/5859.png'

# Loading detector and label binarizer
print('loading detector...')
circle_model = load_model(config.CIRCLE_MODEL_PATH)
label_model = load_model(config.LABEL_MODEL_PATH)
lb = pickle.loads(open(config.LB_PATH, 'rb').read())

# Load the input image from disk and preprocess it, scaling the pixel to the range [0, 1]
image = load_img(imagePaths, target_size=(256, 256))
image = img_to_array(image) / 255.0
image = np.expand_dims(image, axis=0)

# Predict the bounding box of the object along with the class label
(_, labelPreds) = label_model.predict(image)
(innerPreds, outerPreds, _) = circle_model.predict(image)
# (startX, startY, endX, endY) = boxPreds[0]
(innerX, innerY, innerRadius) = innerPreds[0]
(outerX, outerY, outerRadius) = outerPreds[0]

# Determine the class label with the largest predicted probability
i = np.argmax(labelPreds, axis=1)
label = lb.classes_[i][0]

# Load the input image, resize it such that it fits on screen, and grab its dimensions
image = cv2.imread(imagePaths)
image = imutils.resize(image, 720)
(h, w) = image.shape[:2]

# Scale the predicted bounding box coordinates based on the image dimensions
# startX = int(startX * w)
# startY = int(startY * h)
# endX = int(endX * w)
# endY = int(endY * h)
innerX = int(innerX * w)
innerY = int(innerY * h)
innerRadius = int(innerRadius * h)

outerX = int(outerX * w)
outerY = int(outerY * h)
outerRadius = int(outerRadius * h)

concentricity = calculate_concentricity(innerX, innerY, innerRadius, outerX, outerY, outerRadius)

# Draw the predicted bounding box and class label on the image
# inner_y = (innerY - 10) if (innerY - 10) > 10 else innerY + 10
# outer_y = (outerY - 10) if (outerY - 10) > 10 else outerY + 10

if label == 'ok':
    cv2.putText(image, 'QC: {}'.format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 255, 0), 2)
    cv2.putText(image, 'Concentricity: {:.2f}'.format(concentricity), (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 0), 2)
    cv2.circle(image, (innerX, innerY), innerRadius, (0, 255, 0), 2)
    cv2.circle(image, (outerX, outerY), outerRadius, (0, 255, 0), 2)
#     cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
else:
    cv2.putText(image, 'QC: {}'.format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 255, 0), 2)
    cv2.putText(image, 'Concentricity: {:.2f}'.format(concentricity), (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 0),
                2)
    cv2.circle(image, (innerX, innerY), innerRadius, (0, 255, 0), 2)
    cv2.circle(image, (outerX, outerY), outerRadius, (0, 255, 0), 2)
#     cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

# Show the result
cv2.imshow("Result", image)
cv2.waitKey(0)