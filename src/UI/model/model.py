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


class Model:
    save_dir_path = LearningConfig.folder_models_data

    def __init__(self, name: str = "Unnamed", author: str = "Unknown", comment: str = "",
                 created_time: str = get_time(), created_date: str = get_date(),
                 learning_time: str = 0, algorithm="N/A", clf_path: str = None,
                 accuracy=0, threshold=0.5):

        self.name = name
        self.created = f"{created_date} {created_time}"
        self.author = author
        self.comment = comment

        self.clf_path = os.path.join(self.save_dir_path, f"{name}.h5")

        self.algorithm = algorithm
        self.feature_number = None
        self.max_seq_length = None
        self.epoch_number = None
        self.batch_size = None
        self.max_video_frames = None
        self.frame_size = None

        self.threshold = threshold
        self.learning_time = learning_time
        self.accuracy = accuracy

        self.videos_number = None
        self.classes_number = None
        self.test_set_size = None
        self.action_category_list = None

    def begin_learning(self, algorithm, weight, gamma, n_neighbor=None):
        learningTimeStart = time.time()
        self.algorithm = algorithm

        if algorithm == LearningConfig.algorithm_knn:
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
            return (True, "")
        else:
            return (succes_learned, title_warn)
