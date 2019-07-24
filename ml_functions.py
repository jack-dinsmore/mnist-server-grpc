from keras.models import load_model
import tensorflow as tf
import keras.backend as K

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import time

def predict(data, results=None, times=None, id=None):
    start_time = time.time()
    model = load_model("mnist-cnn.h5")
    with tf.device('/gpu:0'):
        predictions = model.predict(data).astype(float)
    K.clear_session()
    predict_time = time.time() - start_time
    if results is not None and id is not None:
        results[id] = predictions
        times[id] = predict_time
    else:
        return predictions, predict_time
