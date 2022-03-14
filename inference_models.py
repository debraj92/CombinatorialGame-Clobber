import time

import tensorflow as tf

from clobber_1d import Clobber_1d
import numpy as np

with tf.io.gfile.GFile("./frozen_models/clobber-black-cnn-frozen.pb", "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    loaded = graph_def.ParseFromString(f.read())


def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    print("-" * 50)
    print("Frozen model layers: ")
    layers = [op.name for op in import_graph.get_operations()]
    if print_graph:
        for layer in layers:
            print(layer)
    print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


# Wrap frozen graph to ConcreteFunctions
frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                inputs=["inputs:0"],
                                outputs=["Identity:0"],
                                print_graph=True)

clobber = Clobber_1d("WBWBWBWWWBWBW", 1, 1)  # Exp : White loses(0) [1 0]
X = clobber.board_features
X = np.reshape(X, (1, 40, 2))
start = time.time()
input_tensor = tf.convert_to_tensor(X)
y = frozen_func(input_tensor)
print(y)
print(y[0].numpy())
end = time.time()
d = end - start
print("time taken ", d)