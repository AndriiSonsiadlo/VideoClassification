# Copyright (C) 2021 Andrii Sonsiadlo

import math
import os
import os.path
import pickle
from concurrent.futures.thread import ThreadPoolExecutor

import cv2
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from sklearn import metrics
from sklearn import svm

from config import DatasetConfig, CustomizationConfig


class SVM_classifier:
	train_dir = DatasetConfig.folder_persons_data
	folder_photo = DatasetConfig.folder_person_photo
	identificated_name = ""
	counter_frame = 0

	def __init__(self, model, path_model, gamma=None):
		self.train_persons = []
		self.test_persons = []
		self.count_train_persons = 0
		self.count_test_persons = 0

		self.Z_test = []
		self.Z_train = []

		self.svm_clf = None
		if (gamma == "auto" or gamma == "scale"):
			self.gamma = gamma
		else:
			self.gamma = "scale"
		self.accuracy = 0

		self.verbose = True

		self.model_full_path = path_model
		self.model = model

	def parting(self, xs, parts):
		part_len = math.ceil(len(xs) / parts)
		return [xs[part_len * k:part_len * (k + 1)] for k in range(parts)]

	def load_image(self, person_names):
		for name in person_names:
			is_trained = False
			is_tested = False
			dir_path_person_photo = os.path.join(self.train_dir, name, self.folder_photo)

			for index, img_path in enumerate(image_files_in_folder(dir_path_person_photo)):

				image = face_recognition.load_image_file(img_path)
				face_bounding_boxes = face_recognition.face_locations(image)

				if len(face_bounding_boxes) != 1:
					# If there are no people (or too many people) in a training image, skip the image.
					if self.verbose:
						print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(
							face_bounding_boxes) < 1 else "Found more than one face"))
				else:
					# Add face encoding for current image to the training set
					encoded = face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0]

					if (index > 5) or (index == 1):
						self.Z_test.append([encoded, name])
						print(f"Added to test - {name}")
						is_tested = True
						continue

					self.Z_train.append([encoded, name])
					is_trained = True
			if not is_trained:
				self.train_persons.remove(name)
			if is_tested:
				self.test_persons.append(name)
		return 1

	def load_model(self):
		try:
			with open(self.model_full_path, 'rb') as f:
				self.svm_clf = pickle.load(f)
			return True
		except BaseException:
			print(f"Error: Model have not loaded\nPath: {self.model_full_path}")
			return False

	def pre_train_svm(self):
		"""
		Trains a k-nearest neighbors classifier for face recognition.

		:param train_dir: directory that contains a sub-directory for each known person, with its name.

		 (View in source code to see train_dir example tree structure)

		 Structure:
			<train_dir>/
			├── <person1>/
			│   ├── <somename1>.jpeg
			│   ├── <somename2>.jpeg
			│   ├── ...
			├── <person2>/
			│   ├── <somename1>.jpeg
			│   └── <somename2>.jpeg
			└── ...

		:param model_save_path: (optional) path to save model on disk
		:param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
		:param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
		:param verbose: verbosity of training
		:return: returns knn classifier that was trained on the given data.
		"""

		# Loop through each person in the training set
		for person_name in os.listdir(self.train_dir):
			dir_path_person_photo = os.path.join(self.train_dir, person_name, self.folder_photo)
			if not os.path.isdir(dir_path_person_photo) or person_name == "temp":
				continue
			else:
				self.train_persons.append(person_name)

		parts_person_names = self.parting(self.train_persons, parts=8)
		print(parts_person_names)
		executor = ThreadPoolExecutor(max_workers=8)
		future1 = executor.submit(self.load_image, parts_person_names[0])
		future2 = executor.submit(self.load_image, parts_person_names[1])
		future3 = executor.submit(self.load_image, parts_person_names[2])
		future4 = executor.submit(self.load_image, parts_person_names[3])
		future5 = executor.submit(self.load_image, parts_person_names[4])
		future6 = executor.submit(self.load_image, parts_person_names[5])
		future7 = executor.submit(self.load_image, parts_person_names[6])
		future8 = executor.submit(self.load_image, parts_person_names[7])
		try:
			test = future1.result() + future2.result() + future3.result() + future4.result() + future5.result() + future6.result() + future7.result() + future8.result()
		except BaseException:
			pass
		self.count_train_persons = len(self.train_persons)
		self.count_test_persons = len(self.test_persons)

	def train(self):
		self.pre_train_svm()
		if (self.count_train_persons):

			X_train = []
			y_train = []
			for x_i, y_i in self.Z_train:
				X_train.append(x_i)
				y_train.append(y_i)

			if self.Z_train:
				self.svm_clf = svm.SVC(kernel='linear', gamma=self.gamma)
				self.svm_clf.fit(X_train, y_train)
				if self.Z_test:
					self.test_predict()
				else:
					print("No data for testing")
				return self.save_model()
			else:
				return (False, "Not found encodings of persons photos and label with names")
		else:
			return (False, "Not found person with one facial photo in database")

	def test_predict(self):
		X_test = []
		y_test = []

		for x_i, y_i in self.Z_test:
			X_test.append(x_i)
			y_test.append(y_i)

		y_pred = self.svm_clf.predict(X_test)

		all_count_photo = len(self.Z_train) + len(self.Z_test)
		perc_test = (len(self.Z_test) * 100) / all_count_photo
		perc_train = 100 - perc_test

		print(f"Test in %: {perc_train}")
		print(f"Train in %: {perc_test}")

		try:
			# Model Accuracy: how often is the classifier correct?
			print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
			self.accuracy = round(metrics.accuracy_score(y_test, y_pred), 6)
		# Model Precision: what percentage of positive tuples are labeled as such?
		#			print("Precision:", metrics.precision_score(y_test, y_pred))
		# Model Recall: what percentage of positive tuples are labelled as such?
		#			print("Recall:", metrics.recall_score(y_test, y_pred))
		except BaseException:
			pass

	def save_model(self):
		# Save the trained KNN classifier
		if self.model_full_path is not None:
			try:
				with open(self.model_full_path, 'wb') as f:
					pickle.dump(self.svm_clf, f)
				return (True, "")
			except BaseException:
				return (False, "Cannot to save a model svm_model.clf")

	def predict_webcam(self, frame):

		X_face_locations = face_recognition.face_locations(frame)
		# If no faces are found in the image, return an empty result.
		if len(X_face_locations) == 0:
			print("No face found")
			self.counter_frame = 0
			return frame, self.counter_frame, ""

		# Find encodings for faces in the test image
		faces_encodings = face_recognition.face_encodings(frame)

		#		predictions = [(pred, loc) if (pred, loc) else (text_unknown, loc) for pred, loc in
		#		               zip(self.svm_clf.predict(faces_encodings), X_face_locations)]

		predictions = [(pred, loc) if (pred, loc) else (CustomizationConfig.text_unknown, loc) for pred, loc in
		               zip(self.svm_clf.predict(faces_encodings), X_face_locations)]

		return self.show_prediction_labels_webcam(frame=frame, predictions=predictions)

	def show_prediction_labels_webcam(self, frame, predictions):

		for name, (top, right, bottom, left) in predictions:
			# Draw a box around the face using the Pillow module
			cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)
			# cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
			# cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

			# There's a bug in Pillow where it blows up with non-UTF-8 text
			# when using the default bitmap font
			# name = name.encode("UTF-8")
			# font = cv2.FONT_HERSHEY_DUPLEX
			# cv2.putText(frame, name, (left + 6, bottom - 6), 1.0, (255, 255, 255), 1)

			# Draw a label with a name below the face
			cv2.putText(frame, "{}".format(name), (left - 10, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			# draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
			# draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

			if len(predictions) == 1:
				if self.identificated_name == name:
					self.counter_frame += 1
				elif (name == CustomizationConfig.text_unknown):
					pass
				else:
					self.identificated_name = name
					self.counter_frame = 0
			else:
				self.counter_frame = 0

		return frame, self.counter_frame, self.identificated_name

	# image predict
	def predict_image(self, X_img_path):
		"""
		Recognizes faces in given image using a trained KNN classifier

		:param X_img_path: path to image to be recognized
		:param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
		:param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
		:param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
			   of mis-classifying an unknown person as a known one.
		:return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
			For faces of unrecognized persons, the name 'unknown' will be returned.
		"""

		# Load image file and find face locations
		X_img = face_recognition.load_image_file(X_img_path)
		X_face_locations = face_recognition.face_locations(X_img)

		# If no faces are found in the image, return an empty result.
		if len(X_face_locations) == 0:
			return X_img, CustomizationConfig.text_unknown

		# Find encodings for faces in the test image
		faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

		# Predict UI and remove classifications that aren't within the threshold
		# predictions = []
		# for pred, loc in zip(self.svm_clf.predict(faces_encodings), X_face_locations):
		# 	if (pred, loc):
		# 		predictions.append([pred, loc])
		# 	else:
		# 		predictions.append([text_unknown, loc])

		predictions = [(pred, loc) if (pred, loc) else (CustomizationConfig.text_unknown, loc) for pred, loc in
		               zip(self.svm_clf.predict(faces_encodings), X_face_locations)]

		return self.show_prediction_labels_on_image(frame=X_img, predictions=predictions)

	def show_prediction_labels_on_image(self, frame, predictions):
		"""
		Shows the face recognition results visually.

		:param img_path: path to image to be recognized
		:param predictions: results of the predict function
		:return:
		"""
		for name, (top, right, bottom, left) in predictions:
			# Draw a box around the face using the Pillow module
			cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
			# cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
			# cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

			# There's a bug in Pillow where it blows up with non-UTF-8 text
			# when using the default bitmap font
			# name = name.encode("UTF-8")
			# font = cv2.FONT_HERSHEY_DUPLEX
			# cv2.putText(frame, name, (left + 6, bottom - 6), 1.0, (255, 255, 255), 1)

			# Draw a label with a name below the face
			cv2.putText(frame, "{}".format(name), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
		# draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
		# draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

		if (len(predictions) == 1):
			return frame, name
		else:
			return frame, CustomizationConfig.text_unknown
