import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
from tensorflow.keras.models import load_model
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# data parameters
TRAIN_DIR = 'training_images'
VAL_DIR = 'validation_images'
TEST_DIR = 'testing_images'
NUM_TRAIN = 20592
NUM_VAL = 6863
NUM_TEST = 7172
SCALED_WIDTH = 84
SCALED_HEIGHT = 84
BATCH_SIZE = 128

# Create data generators
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator()

# load data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(SCALED_HEIGHT, SCALED_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(SCALED_HEIGHT, SCALED_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical')
    
validation_generator = validation_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(SCALED_HEIGHT, SCALED_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# load model and pruning parameters
model = load_model('keras_baseline/baseline_keras_inception_v3.h5')
epochs = 5
end_step = np.ceil(1.0 * NUM_TRAIN / BATCH_SIZE).astype(np.int32) * epochs
pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                   final_sparsity=0.90,
                                                   begin_step=1,
                                                   end_step=end_step,
                                                   frequency=100)
}

# prune model
pruned_model = sparsity.prune_low_magnitude(model, **pruning_params)
pruned_model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# load callback steps
logdir = tempfile.mkdtemp()
callbacks = [
    sparsity.UpdatePruningStep(),
    sparsity.PruningSummaries(log_dir=logdir, profile_batch=0)
]

# train pruned model
EPOCHS = 5
STEPS_PER_EPOCH = math.ceil(NUM_TRAIN/BATCH_SIZE)
VALIDATION_STEPS = math.floor(NUM_VAL/BATCH_SIZE)
history = pruned_model.fit_generator(
        train_generator,
        epochs=epochs,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=validation_generator,
        callbacks = callbacks,
        validation_steps=VALIDATION_STEPS)

# Remove pruning wrappers
final_model = sparsity.strip_pruning(pruned_model)

final_model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results = final_model.evaluate_generator(test_generator)
print('test loss, test acc:', results)

FINAL_MODEL_FILE = 'pruned_keras_inception_v3.h5'
final_model.save(FINAL_MODEL_FILE)