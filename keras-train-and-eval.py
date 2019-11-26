import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as tK
import tensorflow as tf


#creates color coded plot of confusion matrix
def plot_confusion_matrix(cm,classes,
                          title=None,
                          cmap=plt.cm.Blues):
    

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def main():
    CLASSES = 24
    TRAIN_DIR = 'training_images'
    TEST_DIR = 'testing_images'

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
    model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Augment data
    WIDTH = 28
    HEIGHT = 28
    BATCH_SIZE = 32

    train_datagen = ImageDataGenerator()
    validation_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(HEIGHT, WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical')
        
    validation_generator = validation_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(HEIGHT, WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    # perform transfer learning on model
    EPOCHS = 1
    BATCH_SIZE = 32
    STEPS_PER_EPOCH = 320
    VALIDATION_STEPS = 64
    history = model.fit_generator(
        train_generator,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=validation_generator,
        validation_steps=VALIDATION_STEPS)
    

    #find accuracy (should work)
    Y_pred = model.predict_generator(test_generator, STEP_SIZE_VALID)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Total Accuracy = ',end='')
    print(((y_pred == test_generator.classes).sum())/(test_generator.n))

    #find/plot confusion matrix
    con_mat = np.array(confusion_matrix(test_generator.classes, y_pred))
    letters = np.array(['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y'])
    plot_confusion_matrix(con_mat,letters,title= 'Confusion Matrix for Baseline Model')
    plt.show()
    
    

    #flops of model (may not work since not tested on keras model)
    metaDat = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    flops = tf.profiler.profile(graph=tK.get_session().graph,
                                run_meta=metaDat, cmd='op', options=opts)

    print(flops.total_float_ops)  # Prints the "flops" of the model.

    

main()
