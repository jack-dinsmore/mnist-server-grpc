import tensorflow as tf
from keras.models import load_model
import keras.backend as K

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import time, threading

lock = threading.Lock()

def predict(data, results=None, times=None, job_id=None):
    lock.acquire()
    start_time = time.time()
    print("Pass 2")
    model = load_model("mnist-cnn.h5")
    print("Pass 3")
    with tf.device('/gpu:0'):
        predictions = model.predict(data).astype(float)
    K.clear_session()
    predict_time = time.time() - start_time
    lock.release()
    if results is not None and job_id is not None:
        results[job_id] = predictions
        times[job_id] = predict_time
    else:
        return predictions, predict_time