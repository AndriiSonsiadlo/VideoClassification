# Copyright (C) 2021 Andrii Sonsiadlo

import json
import os
import pickle
import shutil

from UI.model.model import Model
from config import LearningConfig, JsonKeyConfig


class ModelList:
    path = os.path.join(LearningConfig.folder_models_data, LearningConfig.filename_model_list_pkl)

    def __init__(self):
        self.list = self.read_from_file()

    def get_list(self):
        return self.list

    def save_to_file(self):
        with open(self.path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def add_model(self, model):
        self.list.append(model)
        print("Added:", model.name)
        self.save_to_file()

    def read_from_file(self):
        model_list = []
        try:
            with open(self.path, 'rb') as input:
                model_list = pickle.load(input).get_list()
        except IOError:
            pass
        # print("File not accessible")
        return model_list

    def get_selected(self):  # returns selected model if exists
        selected = None
        for model in self.list:
            if model.selected:
                selected = model
        return selected

    def set_selected(self, name):  # sets model as selected
        for model in self.list:
            model.selected = False
        self.find_first(name).selected = True
        self.save_to_file()

    def is_empty(self):
        return not bool(self.list)

    def find_first(self, name):  # finds first model that matches name
        found = None
        for model in self.list:
            if model.name == name:
                found = model
        return found

    def check_name_exists(self, name):
        exists = False
        for m in self.list:
            if m.name == name:
                exists = True
        return exists

    def clear_list(self):
        self.list.clear()
        self.save_to_file()

    def print_list(self):
        for m in self.list:
            print(m.name, m.get_time_created(), m.author, m.comment)

    def delete_model(self, name):
        path_model_data = self.find_first(name).path_model_data
        print(path_model_data)
        try:
            for filename in os.listdir(path_model_data):
                file_path = os.path.join(path_model_data, filename)
                os.remove(file_path)
            os.rmdir(path_model_data)
        except Exception:
            print('Failed to delete files')

        name = self.find_first(name).name
        self.list.remove(self.find_first(name))
        self.save_to_file()
        print("Removed:", name)

    def edit_model_name(self, name: str, new_name: str):
        if (self.find_first(name) is not None):
            if self.check_name_exists(new_name):
                print("Name already exists")
            else:
                json_path = os.path.join(self.find_first(name).path_model_data, LearningConfig.file_model_json)
                model_data_dir = self.find_first(name).path_model_data
                name_index = model_data_dir.rfind(name)
                base_dir = model_data_dir[:name_index]

                self.find_first(name).name = new_name
                with open(json_path, "r") as f:
                    data = json.load(f)

                data[JsonKeyConfig.model_name] = new_name

                with open(json_path, 'w') as f:
                    json.dump(data, f)
                    json.dumps(data, indent=4)
                print('RENAME from', model_data_dir, 'TO', base_dir + new_name)
                os.rename(model_data_dir, base_dir + new_name)
                self.find_first(new_name).path_model_data = base_dir + new_name
                self.save_to_file()

    def edit_model_threshold(self, name: str, new_threshold: float):
        if (self.find_first(name) is not None):
            json_path = os.path.join(self.find_first(name).path_model_data, LearningConfig.file_model_json)
            self.find_first(name).threshold = new_threshold
            print(new_threshold)
            with open(json_path, "r") as f:
                data = json.load(f)
            data[JsonKeyConfig.threshold] = new_threshold
            with open(json_path, 'w') as f:
                json.dump(data, f)
                json.dumps(data, indent=4)
            self.save_to_file()

    def edit_model_description(self, name: str, new_desc: str):
        if (self.find_first(name) is not None):
            json_path = os.path.join(self.find_first(name).path_model_data, LearningConfig.file_model_json)

            self.find_first(name).comment = new_desc
            with open(json_path, "r") as f:
                data = json.load(f)

            data[JsonKeyConfig.comment] = new_desc

            with open(json_path, 'w') as f:
                json.dump(data, f)
                json.dumps(data, indent=4)

            self.save_to_file()

    def update_model_list(self):  # check directory names in 'model data' directory update the model list
        folder_model_data = LearningConfig.folder_models_data

        if os.path.isdir(folder_model_data):
            self.clear_list()
            for dir in os.listdir(folder_model_data):
                is_dir_model = os.path.isdir(os.path.join(folder_model_data, dir))
                if is_dir_model:
                    model_name = dir
                    print('name', model_name)

                    # if os.path.isdir(os.path.join(folder_model_data, dir, filename_knn_model)) or os.path.isdir(
                    # 		os.path.join(folder_model_data, dir, filename_svm_model)):
                    try:
                        with open(os.path.join(folder_model_data, model_name, LearningConfig.file_model_json), "r") as read_file:
                            model_data = json.load(read_file)
                            # print(model_data)

                            new_model = Model(name=model_data[model_name],
                                              author=model_data[JsonKeyConfig.author],
                                              comment=model_data[JsonKeyConfig.comment],
                                              created_time=model_data[JsonKeyConfig.p_time],
                                              created_date=model_data[JsonKeyConfig.p_date],
                                              learning_time=model_data[JsonKeyConfig.learning_time],
                                              n_neighbor=model_data[JsonKeyConfig.n_neighbor],
                                              weight=model_data[JsonKeyConfig.weights],
                                              gamma=model_data[JsonKeyConfig.gamma],
                                              algorithm=model_data[JsonKeyConfig.algorithm],
                                              accuracy=model_data[JsonKeyConfig.accuracy],
                                              train_dataset_Y=model_data[JsonKeyConfig.train_dataset],
                                              count_train_Y=model_data[JsonKeyConfig.count_train_dataset],
                                              count_test_Y=model_data[JsonKeyConfig.count_test_dataset],
                                              threshold=model_data[JsonKeyConfig.threshold])
                            self.add_model(new_model)
                    except IOError:
                        print('Model name:', model_name, 'error. No JSON file')
                        try:
                            shutil.rmtree(f"{LearningConfig.folder_models_data}//{model_name}")
                        except BaseException:
                            pass
