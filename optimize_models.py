import keras.models
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tensorflow as tf
from tensorflow.python.tools import freeze_graph

model = keras.models.load_model('/Users/debrajray/MyComputer/clobber/clobber-white-cnn.h5')

# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda inputs: model(inputs))
full_model = full_model.get_concrete_function([tf.TensorSpec(model_input.shape, model_input.dtype) for model_input in model.inputs])
# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()
# Save frozen graph from frozen ConcreteFunction to hard drive
tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir="./frozen_models", name="clobber-white-cnn.pb", as_text=False)
#tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir="./frozen_models", name="clobber-white-cnn.txt", as_text=True)

# python3 -m tensorflow.python.tools.optimize_for_inference --input frozen_models/clobber-white-cnn.pb --output frozen_models/clobber-white-cnn-frozen.pb --frozen_graph=True --input_names=inputs --output_names=Identity

model = keras.models.load_model('/Users/debrajray/MyComputer/clobber/clobber-black-cnn.h5')

# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda inputs: model(inputs))
full_model = full_model.get_concrete_function([tf.TensorSpec(model_input.shape, model_input.dtype) for model_input in model.inputs])
# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()
# Save frozen graph from frozen ConcreteFunction to hard drive
tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir="./frozen_models", name="clobber-black-cnn.pb", as_text=False)

#python3 -m tensorflow.python.tools.optimize_for_inference --input frozen_models/clobber-black-cnn.pb --output frozen_models/clobber-black-cnn-frozen.pb --frozen_graph=True --input_names=inputs --output_names=Identity

