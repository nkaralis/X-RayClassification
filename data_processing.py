"""
	Author: Nikos Karalis
	Big Data Project, Chest XRay Classification using TensorFlow
	Dataset is available at: https://www.kaggle.com/nih-chest-xrays/data
"""

import pandas as pd
import numpy as np
import cv2

IMG_BASE_DIR = "./data/images/"
TRAIN_CSV = "./data/train_labels.csv"
VALIDATION_CSV = "./data/validation_labels.csv"
TEST_CSV = "./data/test_labels.csv"
IMG_RESIZE = 256

def read_train_validation_data():
	df_train = pd.read_csv(TRAIN_CSV, ",")
	df_train['Image Path'] = IMG_BASE_DIR  + df_train["Image Index"]
	df_validation = pd.read_csv(VALIDATION_CSV, ",")
	df_validation['Image Path'] = IMG_BASE_DIR  + df_validation["Image Index"]

	train_imgs = [] 
	train_labels = []
	validation_imgs = []
	validation_labels = []
	for index, row in df_train.iterrows():
		train_imgs.append(cv2.resize(cv2.imread(row["Image Path"], cv2.IMREAD_GRAYSCALE), (IMG_RESIZE, IMG_RESIZE)))
		train_labels.append(row["Finding Labels"])
	for index, row in df_validation.iterrows():
		validation_imgs.append(cv2.resize(cv2.imread(row["Image Path"], cv2.IMREAD_GRAYSCALE), (IMG_RESIZE, IMG_RESIZE)))
		validation_labels.append(row["Finding Labels"])
	# transform labels from strings to integers using numpy
	_, train_labels_np = np.unique(np.array(train_labels), return_inverse=True)
	_, validation_labels_np = np.unique(np.array(validation_labels), return_inverse=True)

	return np.float32(np.array(train_imgs)), np.int32(train_labels_np), np.float32(np.array(validation_imgs)), np.int32(validation_labels_np)

def read_test_data():
	df_test = pd.read_csv(TEST_CSV, ",")
	df_test['Image Path'] = IMG_BASE_DIR  + df_test["Image Index"]

	test_imgs = [] 
	test_labels = []
	for index, row in df_test.iterrows():
		test_imgs.append(cv2.resize(cv2.imread(row["Image Path"], cv2.IMREAD_GRAYSCALE), (IMG_RESIZE, IMG_RESIZE)))
		test_labels.append(row["Finding Labels"])
	# transform labels from strings to integers using numpy
	_, test_labels_np = np.unique(np.array(test_labels), return_inverse=True)
	return np.float32(np.array(test_imgs)), np.int32(test_labels_np)

#read_test_data()
#train_validation_data()