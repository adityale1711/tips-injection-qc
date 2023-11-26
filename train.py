import os
import cv2
import config
import pickle
import numpy as np
import tensorflow as tf
from model import vgg_16
from imutils import paths
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.utils import load_img, img_to_array, to_categorical

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print('loading dataset...')
data = []
labels = []
# bboxes = []
inner = []
outer = []
imagePaths = []

# loop over all CSV files in the annotations directory
for csvPath in paths.list_files(config.ANNOTS_PATH, validExts=('.csv')):
    # Load the contents of the current CSV annotations file
    rows = open(csvPath).read().strip().split('\n')

    # Loop over the rows
    for row in rows:
        # Break the row into the filename, bounding box coordinates, and class label
        row = row.split(',')
        # (filename, startX, startY, endX, endY, label) = row
        (filename, innerX, innerY, innerRadius, outerX, outerY, outerRadius, label) = row

        # Derive the path to the input image, load the image and grab its dimensions
        imagePath = os.path.sep.join([config.IMAGE_PATH, label, filename])
        image = cv2.imread(imagePath)

        if image is not None:
            (h, w) = image.shape[:2]

            # Scale the bounding box coordinates relative to the spatial dimensions of the input image
            # startX = float(startX) / w
            # startY = float(startY) / h
            # endX = float(endX) / w
            # endY = float(endY) / h

            # Scale the circle coordinates relative to the spatial dimensions of the input image
            innerX = float(innerX) / w
            innerY = float(innerY) / h
            innerRadius = float(innerRadius) / w

            outerX = float(outerX) / w
            outerY = float(outerY) / h
            outerRadius = float(outerRadius) / w

            # Load the image and preprocess it
            image = load_img(imagePath, target_size=(256, 256))
            image = img_to_array(image)

            # Update list of data, class label, bounding boxes and image paths
            data.append(image)
            labels.append(label)
            # bboxes.append((startX, startY, endX, endY))
            inner.append((innerX, innerY, innerRadius))
            outer.append((outerX, outerY, outerRadius))
            imagePaths.append(imagePath)

# Convert the data, class labels, bounding boxes and image paths to numpy arrays, scaling the input pixel intensities
# from the range [0, 255] to [0, 1]
data = np.array(data, dtype='float32') / 255.0
labels = np.array(labels)
# bboxes = np.array(bboxes, dtype='float32')
inner = np.array(inner, dtype='float32')
outer = np.array(outer, dtype='float32')
imagePaths = np.array(imagePaths)

# Perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Only there are only twq labels in the dataset, then use Keras/Tensorflow's utility function as well
if len(lb.classes_) == 2:
    labels = to_categorical(labels)

# Partition the data into training and testing splits using 80% of the data for training
# and the remaining 20% for testing
# split = train_test_split(data, labels, bboxes, imagePaths, test_size=0.2, random_state=42)
split = train_test_split(data, labels, inner, outer, imagePaths, test_size=0.2, random_state=42)

# Unpack the data split
# (trainImages, testImages) = split[:2]
# (trainLabels, testLabels) = split[2:4]
# (trainBBoxes, testBBoxes) = split[4:6]
# (trainPaths, testPaths) = split[6:]

(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainInner, testInner) = split[4:6]
(trainOuter, testOuter) = split[6:8]
(trainPaths, testPaths) = split[8:]

# Write the testing image paths to disk so that can use then when evaluating / testing object detector
print('saving testing image paths...')
f = open(config.TEST_PATH, 'w')
#
f.write('\n'.join(testPaths))
f.close()
#
model = vgg_16(lb)
print(model.summary())

# Construct a dictionary for target training outputs
trainTargets = {
    'class_label': trainLabels,
    'inner_circle': trainInner,
    'outer_circle': trainOuter
}

# Construct a second dictionary, this one for target testing outputs
testTargets = {
    'class_label': testLabels,
    'inner_circle': testInner,
    'outer_circle': testOuter
}

# Train the network for bounding box regression and class label prediction
print('training model...')
H = model.fit(trainImages, trainTargets, validation_data=(testImages, testTargets), batch_size=config.BATCH_SIZE,
              epochs=config.NUM_EPOCHS, verbose=1)

# Serialize the model to disk
print('saving detector model...')
model.save(config.MODEL_PATH, save_format='h5')

# Serialize the label binarizer to disk
print('saving label binarizer...')
f = open(config.LB_PATH, 'wb')

f.write(pickle.dumps(lb))
f.close()

# Plot the total loss, label loss and bounding box loss
lossNames = ['loss', 'class_label_loss', 'inner_circle_loss', 'inner_circle_loss']
N = np.arange(0, config.NUM_EPOCHS)

plt.style.use('ggplot')
(fig, ax) = plt.subplots(4, 1, figsize=(13, 13))

# Loop over the loss names
for (i, l) in enumerate(lossNames):
    # Plot the loss for both the training and validation data
    title = 'Loss for {}'.format(l) if l != 'loss' else 'Total loss'

    ax[i].set_title(title)
    ax[i].set_xlabel('Epochs #')
    ax[i].set_ylabel('Loss')
    ax[i].plot(N, H.history[l], label=l)
    ax[i].plot(N, H.history['val_' + l], label='val_' + l)
    ax[i].legend()

# Save the losses figure and create a new figure for the accurates
plt.tight_layout()
plotPath = os.path.sep.join([config.PLOTS_PATH, 'losses_circle.png'])

plt.savefig(plotPath)
plt.close()

# Create a new figure for the accurates
plt.style.use('ggplot')
plt.figure()
plt.plot(N, H.history['class_label_accuracy'], label='class_label_train_acc')
plt.plot(N, H.history['val_class_label_accuracy'], label='val_class_label_acc')
plt.title('Class Label Accuracy')
plt.xlabel('Epochs #')
plt.ylabel('Accuracy')
plt.legend(loc='lower left')

# Save the accurates plot
plotPath = os.path.sep.join([config.PLOTS_PATH, 'accs_circle.png'])
plt.savefig(plotPath)