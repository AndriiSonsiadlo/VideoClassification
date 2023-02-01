# Copyright (C) 2021 Andrii Sonsiadlo

import json
import os
import pickle
import shutil
from collections import defaultdict

from UI.model.model import Model
from config import LearningConfig, JsonKeyConfig, CustomizationConfig


class ModelList(dict):
    path = os.path.join(LearningConfig.folder_models_data, LearningConfig.filename_model_list_pkl)

    selected_model = None
    def __init__(self):
        super().__init__()
        self.read_from_file()

    def save_to_file(self):
        with open(self.path, 'wb') as output:
            pickle.dump(dict(self.items()), output, pickle.HIGHEST_PROTOCOL)

    def add_model(self, model):
        model.name = self.check_name_exists_and_return_new(model.name)
        self[model.name] = model
        self.save_to_file()

    def read_from_file(self):
        try:
            with open(self.path, 'rb') as data:
                self.update(pickle.load(data))
        except IOError:
            pass
        except EOFError:
            self.clear()

    def get_selected(self):
        return self.selected_model

    def set_selected(self, name):  # sets model as selected
        ModelList.selected_model = self[name]
        self.save_to_file()

    def is_empty(self):
        return not bool(self.items())

    def check_name_exists_and_return_new(self, name):
        if not name:
            new_name = CustomizationConfig.text_unnamed
        else:
            new_name = name
        repeated = 1
        while new_name in self:
            new_name = f"{name}({repeated})"
            repeated += 1
        return new_name

    def clear_list(self):
        self.clear()
        self.save_to_file()

    def print_list(self):
        for m in self.values():
            print(m.name, m.get_time_created(), m.author, m.comment)

    def delete_model(self, name):
        clf_path = self[name].clf_path
        try:
            os.remove(clf_path)
        except Exception:
            print('Failed deleting CLF file')
        del self[name]
        print("Removed model:", name)
        self.save_to_file()

    def edit_model(self, model: Model, new_name: str, description: str, threshold: str):
        if model.name != new_name:
            model = self.pop(model.name)
            model.name = self.check_name_exists_and_return_new(new_name)
            self.add_model(model)
        model.description = description
        model.threshold = threshold
        self.save_to_file()

    def check_clf_exists(self):
        for name, model in self:
            if isinstance(model.clf_path, str):
                if not os.path.exists(model.clf_path):
                    model.clf_path = None
