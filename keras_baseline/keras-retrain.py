import numpy as np
import math
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def main():
    CLASSES = 24
    TRAIN_DIR = 'training_images'
    VALIDATION_DIR = 'validation_images'
    TEST_DIR = 'testing_images'
    NUM_TRAIN = 20592
    NUM_VAL = 6863
    NUM_TEST = 7172

    # setup InceptionV3 model and weights while leaving out top layer
    base_model = InceptionV3(weights='imagenet', include_top=False)
    # Add our custom classification layer, preserving the original Inception-v3 architecture
    #     but adapting the output to our number of classes. 
    # We use a GlobalAveragePooling2D preceding the fully-connected Dense layer of 2 outputs.
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # freeze all our base_model layers and train the last ones 
    for layer in base_model.layers:
        layer.trainable = False
        
    # Compile model using an RMSProp optimizer
    #   with the default learning rate of 0.001,
    #   and a categorical_crossentropy — used in multiclass classification tasks — as loss function.
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Rescale data to 84x84, as InceptionV3 requires image sizes at least 75x75
    SCALED_WIDTH = 84
    SCALED_HEIGHT = 84
    BATCH_SIZE = 64

    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Use image generators to rescale images into batches
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(SCALED_HEIGHT, SCALED_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical')
        
    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(SCALED_HEIGHT, SCALED_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(SCALED_HEIGHT, SCALED_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    # perform transfer learning on model
    EPOCHS = 8
    STEPS_PER_EPOCH = math.ceil(NUM_TRAIN/BATCH_SIZE)
    VALIDATION_STEPS = math.floor(NUM_VAL/BATCH_SIZE)
    history = model.fit_generator(
        generator=train_generator,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=validation_generator,
        validation_steps=VALIDATION_STEPS)
    
    # Evaluate the model on the test data using `evaluate`
    print('\n# Evaluate on test data')
    results = model.evaluate_generator(test_generator)
    print('test loss, test acc:', results)

    # save model 
    MODEL_FILE = 'baseline_keras_inception_v3.h5'
    model.save(MODEL_FILE)

main()
