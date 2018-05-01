"""
	Author: Nikos Karalis
	Big Data Project, Chest XRay Classification using TensorFlow
	Dataset is available at: https://www.kaggle.com/nih-chest-xrays/data
"""

import pandas as pd
import numpy as np
import cv2

IMG_BASE_DIR = "./data/random_sample/images/"
CSV_FILE = "./data/random_sample/labels.csv"
IMG_RESIZE = 256
k = 3

def read_data():
	df = pd.read_csv(CSV_FILE, ",")
	df['Image Path'] = IMG_BASE_DIR  + df["Image Index"] 
	# for each image keep: 1) its path, 2) its labels
	df = df.filter(items=["Image Path", "Finding Labels"])
	# keep images that have only one label
	df = df[~df["Finding Labels"].str.contains("No Finding")]
	df = df[~df["Finding Labels"].str.contains("\W")]
	# keep images of top-k labels
	topk_labels = list(df["Finding Labels"].value_counts().nlargest(k).index)
	df = df[df["Finding Labels"].isin(topk_labels)]
	# create training  and validation sets
	df = df.sample(frac=1)
	df_train = df.head(700)
	df_validation = df.tail(198)

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

read_data()