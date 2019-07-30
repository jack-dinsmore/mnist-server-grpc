import os, sys, time, threading
import tensorflow as tf
import numpy as np

# Disable depreciation warnings and limit verbosity during training
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

tf.logging.set_verbosity(0)



NUM_CLASSES = 10
NUM_EPOCHS = 1
IMG_EDGE = 28
MODEL_DIR = 'gs://harrisgroup-ctpu/jtdinsmo/mnist-server/output/'
DATA_DIR = 'gs://harrisgroup-ctpu/jtdinsmo/mnist-server/data/'
TPU_NAME='jtdinsmo-tpu-2'
ZONE_NAME='us-central1-b'
PROJECT_NAME = 'harrisgroup-223921'
NUM_ITERATIONS = 50 # Number of iterations per TPU training loop
TRAIN_STEPS = 100000
EVALUATE_STEPS = 1000
INFERENCE_TIME_THRESHOLD = 10 # Seconds
NUM_SHARDS = 8 # Number of shards (TPU chips).
LEARNING_RATE = 1.0
USE_TPU = True

lock = threading.Lock()

# DEFINE THE NETWORK

class ModelCNN(object):
    def __call__(self, inputs):
        net = tf.layers.conv2d(inputs, 32, [5, 5], activation=tf.nn.relu, name='conv1')
        net = tf.layers.max_pooling2d(net, [2, 2], 2, name='pool1')
        net = tf.layers.conv2d(net, 64, [5, 5], activation=tf.nn.relu, name='conv2')
        net = tf.layers.max_pooling2d(net, [2, 2], 2, name='pool2')
        net = tf.layers.flatten(net, name='flat')
        net = tf.layers.dense(net, NUM_CLASSES, activation=None, name='fc1')
        return net

# DEFINE THE INPUT FUNCTIONS

def metric_fn(labels, logits):
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=tf.argmax(logits, axis=1))
    return {"accuracy": accuracy}

def model_fn(features, labels, mode, params):
    del params# Unused
    image = features
    if isinstance(image, dict):
        image = features["image"]

    image = tf.reshape(image, [-1, IMG_EDGE, IMG_EDGE, 1])
    model = ModelCNN()
    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model(image)
        predictions = {
            'class_ids': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits),
        }
        return tf.contrib.tpu.TPUEstimatorSpec(mode, predictions=predictions)

    logits = model(image)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE,
            tf.train.get_global_step(),
            decay_steps=100000,
            decay_rate=0.96)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        if USE_TPU:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_global_step()))

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, loss=loss, eval_metrics=(metric_fn, [labels, logits]))


# CREATE AND PREDICT WITH TPUS

def create_estimator(batch_size):
    print("Creating the estimator")
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        TPU_NAME,
        zone=ZONE_NAME,
        project=PROJECT_NAME)

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=MODEL_DIR,
        session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True),
        tpu_config=tf.contrib.tpu.TPUConfig(NUM_ITERATIONS, NUM_SHARDS))

    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        use_tpu=USE_TPU,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        predict_batch_size=batch_size,
        params={"data_dir": DATA_DIR},
        config=run_config)
    return estimator

def predict(data, batch_size, results=None, times=None, job_id=None):
    assert (results is None and times is None and job_id is None) or not (results is None or times is None or job_id is None)
    lock.acquire()
    start_time = time.time()

    estimator = create_estimator(batch_size)

    print("Predicting")
    def predict_input_fn(params):
        batch_size = params["batch_size"]
        dataset_predict = tf.data.Dataset.from_tensor_slices(data.astype('float32'))
        return dataset_predict.batch(batch_size)

    predictions = estimator.predict(predict_input_fn)

    print("Getting labels")
    labels = []
    for pred_dict in predictions:
        labels.append(pred_dict['probabilities'])
    labels = np.array(labels).astype('float64')
    tf.reset_default_graph()

    predict_time = time.time() - start_time
    lock.release()

    print("Returned")
    if results is not None:
        results[job_id] = labels
        times[job_id] = predict_time
    return labels, predict_time

