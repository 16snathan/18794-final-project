import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
from tensorflow.keras.models import load_model
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# data parameters
TRAIN_DIR = 'training_images'
TEST_DIR = 'testing_images'
WIDTH = 28
HEIGHT = 28
BATCH_SIZE = 32
train_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator()

# load data
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

# load model and pruning parameters
model = load_model('baseline_keras_inception_v3.h5')
num_train_samples = 27455
batch_size = 32
epochs = 12
end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * epochs
pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                   final_sparsity=0.90,
                                                   begin_step=2000,
                                                   end_step=end_step,
                                                   frequency=100)
}

# prune model
pruned_model = sparsity.prune_low_magnitude(model, **pruning_params)
pruned_model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# train pruned model
logdir = tempfile.mkdtemp()
callbacks = [
    sparsity.UpdatePruningStep(),
    sparsity.PruningSummaries(log_dir=logdir, profile_batch=0)
]

STEPS_PER_EPOCH = 320
VALIDATION_STEPS = 64
pruned_model.fit_generator(
        train_generator,
        epochs=epochs,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=validation_generator,
        callbacks = callbacks,
        validation_steps=VALIDATION_STEPS)

# Remove pruning wrappers
model = sparsity.strip_pruning(pruned_model)

#FINAL_MODEL_FILE = 'pruned_keras_inception_v3.h5'
#final_model.save(FINAL_MODEL_FILE)


#find accuracy (should work)
Y_pred = model.predict_generator(validation_generator, VALIDATION_STEPS+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Total Accuracy = ',end='')
print(((y_pred == validation_generator.classes).sum())/(validation_generator.n))

#create/plot confusion matrix
con_mat = np.array(confusion_matrix(validation_generator.classes, y_pred))
letters = np.array(['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y'])
plot_confusion_matrix(con_mat,letters,title= 'Confusion Matrix for Baseline Model')
plt.show()



#flops of model (may not work since not tested on keras model)
metaDat = tf.RunMetadata()
opts = tf.profiler.ProfileOptionBuilder.float_operation()

flops = tf.profiler.profile(graph=tK.get_session().graph,
                            run_meta=metaDat, cmd='op', options=opts)

print(flops.total_float_ops)



