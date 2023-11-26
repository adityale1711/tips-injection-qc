import os

# Define the base path to the input dataset and the use it to derive the path
# to the input images and annotations CSV files
BASE_PATH = 'datasets'
IMAGE_PATH = os.path.sep.join([BASE_PATH, 'images'])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, 'annotations'])

# Define the path to the base output directory
BASE_OUTPUT = 'output'

# Define the path to the output model, label binarizer, plots output directory and testing image paths
CIRCLE_MODEL_PATH = os.path.sep.join([BASE_OUTPUT, 'circle_detector_model.h5'])
LABEL_MODEL_PATH = os.path.sep.join([BASE_OUTPUT, 'detector_model.h5'])
LB_PATH = os.path.sep.join([BASE_OUTPUT, 'lb.pickle'])
PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, 'plots'])
TEST_PATH = os.path.sep.join([BASE_OUTPUT, 'test_paths.txt'])

# Initialize learning rate, number of epochs to train for and batch size
LR = 1e-4
NUM_EPOCHS = 20
BATCH_SIZE = 2