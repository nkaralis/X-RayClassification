"""
	Author: Nikos Karalis
	Big Data Project, Chest XRay Classification using TensorFlow
	Dataset is available at: https://www.kaggle.com/nih-chest-xrays/data
	Source code is based on the tutorial provided by TensorFlow: https://www.tensorflow.org/tutorials/layers
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

from data_processing import read_train_validation_data
from model import cnn_model_fn

TEST_EPOCHS = 1
TRAIN_EPOCHS = 10
BATCH_SIZE = 10
STEPS = None

def main(unused_arg):
  # Load training and eval data
  train_data, train_labels, eval_data, eval_labels = read_train_validation_data()
  # Create the Estimator
  xray_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="./models/4conv_2")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=500)
  
  # Train the model
  start_time = time.time()
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=BATCH_SIZE,
      num_epochs=TRAIN_EPOCHS,
      shuffle=False)
  xray_classifier.train(
      input_fn=train_input_fn,
      steps=STEPS,
      hooks=[logging_hook])
  print("--- %s seconds ---" % (time.time() - start_time))

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      batch_size=BATCH_SIZE,
      num_epochs=TEST_EPOCHS,
      shuffle=False)
  eval_results = xray_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

if __name__ == "__main__":
	tf.app.run()