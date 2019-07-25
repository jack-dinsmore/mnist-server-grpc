import tensorflow as tf
from keras.models import load_model
import keras.backend as K

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import time, threading
import numpy as np

lock = threading.Lock()


USE_TPU=True
NUM_CLASSES = 10
NUM_EPOCHS = 5
IMG_EDGE = 28
MODEL_DIR = 'gs://harrisgroup-ctpu/jtdinsmo/mnist-server/output/'
DATA_DIR = 'gs://harrisgroup-ctpu/jtdinsmo/mnist-server/data/'
TPU_NAME='jtdinsmo-tpu-server'
ZONE_NAME='us-central1-b'
PROJECT_NAME = 'harrisgroup-223921'
NUM_ITERATIONS = 50 # Number of iterations per TPU training loop
TRAIN_STEPS = 100000
EVALUATE_STEPS = 1000
INFERENCE_TIME_THRESHOLD = 10 # Seconds
NUM_SHARDS = 8 # Number of shards (TPU chips).
LEARNING_RATE = 1
BATCH_SIZE = 128

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images  # Returns an np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images  # Returns an np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

def metric_fn(labels, logits):
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=tf.argmax(logits, axis=1))
    return {"accuracy": accuracy}
    
def train_input_fn(params):
    batch_size = params["batch_size"]
    train_data_dataset = tf.data.Dataset.from_tensor_slices(train_data)
    train_labels_dataset = tf.data.Dataset.from_tensor_slices(train_labels)
    single_train = tf.data.Dataset.zip((train_data_dataset, train_labels_dataset))
    dataset_train = single_train
    # Create epochs the dumb way: just keep adding shuffled versions of the same data onto the dataset
    for _ in range(NUM_EPOCHS-1):
        dataset_train = dataset_train.concatenate(single_train.shuffle(train_data.shape[0]))
    return dataset_train.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

class Model(object):
    def __call__(self, inputs):
        net = tf.layers.conv2d(inputs, 32, [5, 5], activation=tf.nn.relu, name='conv1')
        net = tf.layers.max_pooling2d(net, [2, 2], 2, name='pool1')
        net = tf.layers.conv2d(net, 64, [5, 5], activation=tf.nn.relu, name='conv2')
        net = tf.layers.max_pooling2d(net, [2, 2], 2, name='pool2')
        net = tf.layers.flatten(net, name='flat')
        net = tf.layers.dense(net, NUM_CLASSES, activation=None, name='fc1')
        return net
    
def model_fn(self, features, labels, mode, params):
    del params# Unused
    image = features
    if isinstance(image, dict):
        image = features["image"]
        
    model = Model()
    image = tf.reshape(features, [-1, IMG_EDGE, IMG_EDGE, 1])

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
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
        if USE_TPU:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_global_step()))

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, loss=loss, eval_metrics=(metric_fn, [labels, logits]))


def get_predictions(data):
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        TPU_NAME,
        zone=ZONE_NAME,
        project=PROJECT_NAME)

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=MODEL_DIR,
        session_config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=True),
        tpu_config=tf.contrib.tpu.TPUConfig(NUM_ITERATIONS, NUM_SHARDS),)
    
    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        use_tpu=USE_TPU,
        train_batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
        predict_batch_size=BATCH_SIZE,
        params={"data_dir": DATA_DIR},
        config=run_config)

    print("Training")
    evaluate_steps = train_data.shape[0] // BATCH_SIZE
    train_steps = NUM_EPOCHS * evaluate_steps
    try:
        self.estimator.train(input_fn=train_input_fn, steps=train_steps)
    except tf.errors.OutOfRangeError:
        pass
        
    print("Predicting")
    def predict_input_fn(params):
        batch_size = params["batch_size"]
        dataset_predict = tf.data.Dataset.from_tensor_slices(data)
        return dataset_predict.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        
    predictions = self.estimator.predict(predict_input_fn)
    
    labels = []
    for pred_dict in predictions:
        print(pred_dict['class_ids'], pred_dict['probabilities'])
        labels.append(pred_dict['class_ids'])
    labels = np.array(labels)
    
    return labels


def predict(data, results=None, times=None, job_id=None):
    lock.acquire()
    start_time = time.time()
    
    predictions = get_predictions(data)

    predict_time = time.time() - start_time
    lock.release()
    if results is not None and job_id is not None:
        results[job_id] = predictions
        times[job_id] = predict_time
    else:
        return predictions, predict_time