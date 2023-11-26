import config
from keras.models import Model
from keras.optimizers import Adam
from keras.applications import VGG16
from keras.layers import Input, Flatten, Dense, Dropout

def vgg_16(lb):
    # load the VGG16 network, ensuring the head FC layers are left off
    vgg = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(256, 256, 3)))

    # Freeze all VGG layers, so they will not be updated during the training process
    vgg.trainable = False

    # Flatten the max-pooling output of VGG
    flatten = vgg.output
    flatten = Flatten()(flatten)

    # Construct a fully-connected layer header to output the predicted bounding box coordinates
    circleLayer = Dense(128, activation='relu')(flatten)
    circleLayer = Dense(64, activation='relu')(circleLayer)
    circleLayer = Dense(32, activation='relu')(circleLayer)

    # Output Circles
    innerCircle = Dense(3, activation='sigmoid', name='inner_circle')(circleLayer)
    outerCircle = Dense(3, activation='sigmoid', name='outer_circle')(circleLayer)

    # Construct a fully-connected layer head, this one to predict the class label
    classLayer = Dense(512, activation='relu')(flatten)
    classLayer = Dropout(0.5)(classLayer)
    classLayer = Dense(512, activation='relu')(classLayer)
    classLayer = Dropout(0.5)(classLayer)
    classLayer = Dense(len(lb.classes_), activation='softmax', name='class_label')(classLayer)

    # Put together which accept an input image and then output bounding box coordinates and a class label
    model = Model(inputs=vgg.input, outputs=(innerCircle, outerCircle, classLayer))

    # Define a dictionary to set the loss methods -- categorical cross-entropy for the class label head
    # and mean absolute error for the bounding box head
    losses = {
        'class_label': 'categorical_crossentropy',
        'inner_circle': 'mean_squared_error',
        'outer_circle': 'mean_squared_error'
    }

    # Define a dictionary that specifies the weights per loss
    # (both the class label and bounding box outputs will receive equal weight)
    lossWeights = {
        'class_label': 1.0,
        'inner_circle': 1.0,
        'outer_circle': 1.0
    }

    # Initialize the optimizer, compile the model and show the model summary
    opt = Adam(learning_rate=config.LR)
    model.compile(loss=losses, optimizer=opt, metrics=['accuracy'], loss_weights=lossWeights)

    return model