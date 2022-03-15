import time

import tensorflow as tf
import numpy as np

from clobber_1d import Clobber_1d

clobber = Clobber_1d("WBWBWBWWWBWBW", 1, 1)  # Exp : White loses(0) [1 0]
clobber.computePrunedMovesFromSubgames(0)
X = clobber.board_features
X = np.reshape(X, (1, 40, 2))

start = time.time()
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="clobber-black-cnn.tflite")
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], X)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
end = time.time()
d = end - start
print("time taken ", d)
print(output_data)

