#### NOTES
#### NEED TO CHANGE DATA BEING LOADED IN
#### CURRENTLY THIS USES THE MNIST DIGIT DATASET
#### NEED TO CHANGE:
#       value of nb_classes
#       data that (X_train, y_train), (X_test, y_test) refer to
#       X_train.reshape
#       Y_train.reshape

# from convert_csv_to_numpy_array import load_image_data
import pickle
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation, Input, Embedding, LSTM, Dense, Lambda, GaussianNoise, concatenate
from tensorflow.keras.models import Model, load_model
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import MaxPooling2D, Dropout, Dense, Flatten, Activation, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy as logloss
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

################################################################
################################################################ 
# LOAD DATA
nb_classes = 25
with open('sign_mnist_numpy_arrays.pickle', 'rb') as f:
    ((X_train, y_train), (X_test, y_test)) = pickle.load(f)
print("Loaded image data!")
# convert y_train and y_test to categorical binary values 
Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)

################################################################
################################################################
# PREPROCESS DATA
# Reshape them to batch_size, width,height, #channels
X_train = X_train.reshape(27455, 28, 28, 1)
X_test = X_test.reshape(7172, 28, 28, 1)
print("Image data reshaped to batch size!")

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalize the values
X_train /= 255
X_test /= 255

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

################################################################
################################################################
# DEFINE TEACHER MODEL

input_shape = (28, 28, 1) # Input shape of each image

# Hyperparameters
nb_filters = 64 # number of convolutional filters to use
pool_size = (2, 2) # size of pooling area for max pooling
kernel_size = (3, 3) # convolution kernel size

teacher = load_model('baseline_keras_inception_v3.h5')

print("SUMMARY OF BASELINE INCEPTION MODEL:")
print(teacher.summary())

################################################################
################################################################
# DEFINE STUDENT MODEL
student = Sequential()
student.add(Flatten(input_shape=input_shape))
student.add(Dense(32, activation='relu'))
student.add(Dropout(0.2))
student.add(Dense(nb_classes))
student.add(Activation('softmax'))

#sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
student.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

student.summary()

################################################################
################################################################
# Define a new model that outputs only teacher logits
# Raise the temperature of teacher model and gather the soft targets
# Set a tempature value
temp = 7

#Collect the logits from the previous layer output and store it in a different model
teacher_WO_Softmax = Model(teacher.input, teacher.get_layer('dense').output)

################################################################
################################################################
# Define a manual softmax function
def softmax(x):
    return np.exp(x)/(np.exp(x).sum())

################################################################
################################################################
# For example, just grab the first image and lets see how softening of probabilities work
intermediate_output = teacher_WO_Softmax.predict(X_test[0].reshape(1,28,28,1))
print(softmax(intermediate_output))

# logits for the first number in test dataset
x = intermediate_output[0]

################################################################
################################################################
# Prepare the soft targets and the target data for student to be trained upon
teacher_train_logits = teacher_WO_Softmax.predict(X_train)
# This model directly gives the logits (see the teacher_WO_softmax model above)
teacher_test_logits = teacher_WO_Softmax.predict(X_test) 

# Perform a manual softmax at raised temperature
train_logits_T = teacher_train_logits/temp
test_logits_T = teacher_test_logits / temp 

Y_train_soft = softmax(train_logits_T)
Y_test_soft = softmax(test_logits_T)

# Concatenate so that this becomes a 10 + 10 dimensional vector
Y_train_new = np.concatenate([Y_train, Y_train_soft], axis=1)
Y_test_new =  np.concatenate([Y_test, Y_test_soft], axis =1)

################################################################
################################################################
# Prepare the student model that outputs probabilities with and without temperature

# Remove the softmax layer from the student network
student.layers.pop()

# Now collect the logits from the last layer
# This is going to be a tensor. And hence it needs to pass through a Activation layer
logits = student.layers[-1].output 
probs = Activation('softmax')(logits)

# softed probabilities at raised temperature
logits_T = Lambda(lambda x: x / temp)(logits)
probs_T = Activation('softmax')(logits_T)

output = concatenate([probs, probs_T])

# This is our new student model
student = Model(student.input, output)

student.summary()

################################################################
################################################################
# This will be a teacher trained student model. 
# --> This uses a knowledge distillation loss function

epochs = 12

# Declare knowledge distillation loss
def knowledge_distillation_loss(y_true, y_pred, alpha):

    # Extract the one-hot encoded values and the softs separately so that we can create two objective functions
    y_true, y_true_softs = y_true[: , :nb_classes], y_true[: , nb_classes:]
    
    y_pred, y_pred_softs = y_pred[: , :nb_classes], y_pred[: , nb_classes:]
    
    loss = alpha*logloss(y_true,y_pred) + logloss(y_true_softs, y_pred_softs)
    
    return loss

# For testing use regular output probabilities - without temperature
def acc(y_true, y_pred):
    y_true = y_true[:, :nb_classes]
    y_pred = y_pred[:, :nb_classes]
    return categorical_accuracy(y_true, y_pred)

student.compile(
    #optimizer=optimizers.SGD(lr=1e-1, momentum=0.9, nesterov=True),
    optimizer='adadelta',
    loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, 0.1),
    #loss='categorical_crossentropy',
    metrics=[acc] )

# Train student model
student.fit(X_train, Y_train_new,
          batch_size=256,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, Y_test_new))

# This is a standalone student model (same number of layers as original student model) trained on same data
# for comparing it with teacher trained student.

n_student = Sequential()
n_student.add(Flatten(input_shape=input_shape))
n_student.add(Dense(32, activation='relu'))
n_student.add(Dropout(0.2))
n_student.add(Dense(nb_classes))
n_student.add(Activation('softmax'))

#sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
n_student.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

n_student.fit(X_train, Y_train,
          batch_size=256,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, Y_test))