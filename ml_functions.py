import tensorflow as tf
from keras.models import load_model
import keras.backend as K

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import time, threading

lock = threading.Lock()

global model, model_loaded
model = None
model_loaded = False

def get_model():
    global model, model_loaded
    if not model_loaded:
        print("LOADING")
        model = load_model("mnist-cnn.h5")
        model_loaded = True
    return model

def predict(data, batch_size, results=None, times=None, job_id=None):
    lock.acquire()
    start_time = time.time()
    model = get_model()
    with tf.device('/gpu:0'):
        predictions = model.predict(data, batch_size=batch_size).astype(float)
    predict_time = time.time() - start_time
    lock.release()
    if results is not None and job_id is not None:
        results[job_id] = predictions
        times[job_id] = predict_time
    else:
        return predictions, predict_time

def cleanup():
    K.clear_session()
