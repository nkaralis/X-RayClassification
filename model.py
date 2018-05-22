"""
	Author: Nikos Karalis
	Big Data Project, Chest XRay Classification using TensorFlow
	Dataset is available at: https://www.kaggle.com/nih-chest-xrays/data
	Source code is based on the tutorial provided by TensorFlow: https://www.tensorflow.org/tutorials/layers
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

IMG_SIZE = 256
COLOR = 1
BATCH_SIZE = -1
LEARINING_RATE = 0.0002

def cnn_model_fn(features, labels, mode):
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # Resized X-Ray images are 256x256 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [BATCH_SIZE, IMG_SIZE, IMG_SIZE, COLOR])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[16, 16],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  # pool1 = tf.layers.average_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[8, 8],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  # pool2 = tf.layers.average_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Convolutional Layer #3
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=16,
      kernel_size=[4, 4],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #3
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
  # pool3 = tf.layers.average_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

  # Convolutional Layer #4
  conv4 = tf.layers.conv2d(
      inputs=pool3,
      filters=8,
      kernel_size=[4, 4],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #3
  pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
  # pool4 = tf.layers.average_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # pool2_flat = tf.reshape(pool2,[-1,32*32*256])
  # pool3_flat = tf.reshape(pool3,[-1,16*16*32])
  pool4_flat = tf.reshape(pool4, [-1, 16*16*8])

  # Dense Layer
  dense = tf.layers.dense(inputs=pool4_flat, units=512, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Output Tensor Shape: [batch_size, 5]
  logits = tf.layers.dense(inputs=dropout, units=5)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARINING_RATE)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"]),
      "recall": tf.metrics.recall(
          labels=labels, predictions=predictions["classes"]),
      "precision": tf.metrics.precision(
          labels=labels, predictions=predictions["classes"])}
  results = tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, predictions=predictions)
  print(results.predictions)
  return results