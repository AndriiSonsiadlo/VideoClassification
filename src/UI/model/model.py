# Copyright (C) 2021 Andrii Sonsiadlo

import json
import os
from datetime import datetime, time, date
import tkinter as tk
import time

from UI.algorithm_det_reg.knn import KNN_classifier
from UI.algorithm_det_reg.svm import SVM_classifier
from UI.get_time import get_time, get_date
from config import LearningConfig, JsonKeyConfig

root = tk.Tk()
root.withdraw()

class Model:
	def __init__(self, name: str = "Unnamed", author: str = "Unknown", comment: str = "",
	             created_time: str = get_time(), created_date: str = get_date(),
	             learning_time: str = 0, n_neighbor=0, algorithm="N/A", weight='N/A', gamma='N/A', train_dataset_Y=[],
	             count_train_Y=0,
	             test_dataset_Y=[], count_test_Y=0, accuracy=0, threshold=0.5):

		self.name = name
		self.created_time = created_time
		self.created_date = created_date
		self.created = f"{self.created_date} {self.created_time}"
		self.author = author
		self.comment = comment
		self.selected = False

		self.algorithm = algorithm
		self.n_neighbor = n_neighbor  # KNN
		self.weight = weight  # KKN
		self.gamma = gamma  # SVM
		self.threshold = threshold
		self.learning_time = learning_time
		self.accuracy = accuracy

		self.count_train_Y = count_train_Y
		self.count_test_Y = count_test_Y
		self.train_dataset_Y = train_dataset_Y
		self.test_dataset_Y = test_dataset_Y

		self.init_paths()

	def init_paths(self):
		self.path_model_data = os.path.join(LearningConfig.folder_models_data, self.name)
		if self.algorithm == LearningConfig.algorithm_knn:
			self.path_file_model = os.path.join(LearningConfig.folder_models_data, self.name, LearningConfig.filename_knn_model)
		else:
			self.path_file_model = os.path.join(LearningConfig.folder_models_data, self.name, LearningConfig.filename_svm_model)

	def edit(self, new_name: str):
		self.name = new_name

	# what should be editable? what about creation date?

	def begin_learning(self, algorithm, weight, gamma, n_neighbor=None):
		learningTimeStart = time.time()
		self.algorithm = algorithm
		self.init_paths()

		if (algorithm == LearningConfig.algorithm_knn):
			algorithm_object = KNN_classifier(self, path_model=self.path_file_model, n_neighbor=n_neighbor,
			                                  weight=weight)
			self.weight = algorithm_object.weight
		else:
			algorithm_object = SVM_classifier(self, path_model=self.path_file_model, gamma=gamma)
			self.gamma = algorithm_object.gamma
		try:
			succes_learned, title_warn = algorithm_object.train()
		except BaseException:
			return (False, "Cannot to train a model")

		if (succes_learned):
			learningTimeStop = time.time() - learningTimeStart
			self.learning_time = round(learningTimeStop, 2)
			self.train_dataset_Y = algorithm_object.train_persons
			self.test_dataset_Y = algorithm_object.test_persons
			self.count_train_Y = algorithm_object.count_train_persons
			self.count_test_Y = algorithm_object.count_test_persons
			self.accuracy = algorithm_object.accuracy

			self.save_to_json()
			return (True, "")
		else:
			return (succes_learned, title_warn)

	def save_to_json(self):
		filepath_model_json = os.path.join(LearningConfig.folder_models_data, self.name, LearningConfig.file_model_json)
		dataJSON = {
			JsonKeyConfig.model_name: self.name,
			JsonKeyConfig.author: self.author,
			JsonKeyConfig.comment: self.comment,
			JsonKeyConfig.p_date: self.created_date,
			JsonKeyConfig.p_time: self.created_time,
			JsonKeyConfig.learning_time: self.learning_time,
			JsonKeyConfig.algorithm: self.algorithm,
			JsonKeyConfig.n_neighbor: self.n_neighbor,
			JsonKeyConfig.gamma: self.gamma,
			JsonKeyConfig.weights: self.weight,
			JsonKeyConfig.threshold: self.threshold,
			JsonKeyConfig.accuracy: self.accuracy,
			# test_size: test_size,
			# train_size: (1.0 - test_size),
			JsonKeyConfig.count_train_dataset: self.count_train_Y,
			JsonKeyConfig.count_test_dataset: self.count_test_Y,
			JsonKeyConfig.train_dataset: self.train_dataset_Y,
			JsonKeyConfig.test_dataset: self.test_dataset_Y
		}
		print("[INFO] saving data of model to .json...")
		with open(filepath_model_json, "w") as write_file:
			json.dump(dataJSON, write_file)
			json.dumps(dataJSON, indent=4)
