"""
	Author: Nikos Karalis
	Big Data Project, Chest XRay Classification using TensorFlow
	Dataset is available at: https://www.kaggle.com/nih-chest-xrays/data
	Source code is based on the tutorial provided by TensorFlow: https://www.tensorflow.org/tutorials/layers
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

from data_processing import read_test_data
from model import cnn_model_fn

BATCH_SIZE = 10
TEST_EPOCHS = 1

def main(unused_arg):
  # Load test data
  test_data, test_labels = read_test_data()
  # Load the Estimator
  xray_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="./models/4conv_2")

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": test_data},
      y=test_labels,
      batch_size=BATCH_SIZE,
      num_epochs=TEST_EPOCHS,
      shuffle=False)
  # eval_results = xray_classifier.evaluate(input_fn=eval_input_fn)
  # print(eval_results)
  eval_results = xray_classifier.predict(input_fn=eval_input_fn)
  i = 0
  for x in eval_results:
  	print("Truth: %s Predicted: %s" % (test_labels[i], x["classes"]))
  	i += 1

if __name__ == "__main__":
	tf.app.run()