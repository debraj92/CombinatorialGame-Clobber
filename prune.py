import keras.models
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from keras.models import save_model
import numpy as np

model = keras.models.load_model('./clobber-black-cnn.h5')
'''
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

total_training_samples = pow(3, 11)
batch_size = total_training_samples
# Finish pruning after 10 epochs
pruning_epochs = 10
num_images = total_training_samples * (0.8)
#end_step = np.ceil(num_images / batch_size).astype(np.int32) * pruning_epochs

# Define pruning configuration
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.40,
                                                               final_sparsity=0.70,
                                                               begin_step=0,
                                                               end_step=pruning_epochs)
}
model_for_pruning = prune_low_magnitude(model, **pruning_params)

# Recompile the model
model_for_pruning.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

# Model callbacks
callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep()
]

with open('train_data_samples-black.npy', 'rb') as f:
    input_train = np.load(f)

with open('labels-black.npy', 'rb') as f:
    target_train = np.load(f)

# Fitting data
model_for_pruning.fit(input_train, target_train,
                      batch_size=batch_size,
                      epochs=pruning_epochs,
                      callbacks=callbacks,
                      validation_split=0.2)

# Export the model
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
#save_model(model_for_export, "./clobber-black-cnn-pruned.h5")
'''
model_for_export = model
#TF-LITE
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_and_pruned_tflite_model = converter.convert()

with open('./clobber-black-cnn.tflite', 'wb') as f:
    f.write(quantized_and_pruned_tflite_model)


# Repeat for white


model = keras.models.load_model('./clobber-white-cnn.h5')
'''
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

total_training_samples = pow(3, 11)
batch_size = total_training_samples
# Finish pruning after 10 epochs
pruning_epochs = 10

# Define pruning configuration
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.40,
                                                               final_sparsity=0.70,
                                                               begin_step=0,
                                                               end_step=pruning_epochs)
}
model_for_pruning = prune_low_magnitude(model, **pruning_params)

# Recompile the model
model_for_pruning.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

# Model callbacks
callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep()
]

with open('train_data_samples-white.npy', 'rb') as f:
    input_train = np.load(f)

with open('labels-white.npy', 'rb') as f:
    target_train = np.load(f)

# Fitting data
model_for_pruning.fit(input_train, target_train,
                      batch_size=batch_size,
                      epochs=pruning_epochs,
                      callbacks=callbacks,
                      validation_split=0.2)

# Export the model
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
'''
model_for_export = model
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_and_pruned_tflite_model = converter.convert()

with open('./clobber-white-cnn.tflite', 'wb') as f:
    f.write(quantized_and_pruned_tflite_model)

