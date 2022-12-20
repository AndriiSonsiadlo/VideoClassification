# Copyright (C) 2021 Andrii Sonsiadlo

import json
import os
import pickle
import shutil

from UI.person.person import Person
from config import LearningConfig, DatasetConfig, JsonKeyConfig


class ActionList:
    path = f'{DatasetConfig.folder_persons_data}/{DatasetConfig.file_person_list_pkl}'

    def __init__(self):
        self.list = self.read_from_file()

    def get_list_with_photo(self, num_photos=LearningConfig.default_count_photos):
        list = []
        for person in self.list:
            if len(person.photo_paths) >= num_photos:
                list.append(person)
        return list

    def get_list(self):
        return self.list

    def save_to_file(self):
        with open(self.path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def update_person_list(self):
        if (os.path.isdir(DatasetConfig.folder_persons_data)):
            self.clear_list()
            for folder_person in os.listdir(DatasetConfig.folder_persons_data):
                is_dir_person = os.path.isdir(os.path.join(DatasetConfig.folder_persons_data, folder_person))

                if is_dir_person and folder_person != DatasetConfig.folder_temp:
                    person_name = folder_person
                    try:
                        with open(
                                DatasetConfig.folder_persons_data + '/' + person_name + '/' + DatasetConfig.file_person_json,
                                "r") as read_file:
                            person_data = json.load(read_file)

                            new_person = Person(name=person_data[JsonKeyConfig.information][JsonKeyConfig.p_name],
                                                is_wanted=person_data[JsonKeyConfig.information][
                                                    JsonKeyConfig.p_is_wanted],
                                                age=person_data[JsonKeyConfig.information][JsonKeyConfig.p_age],
                                                gender=person_data[JsonKeyConfig.information][JsonKeyConfig.p_gender],
                                                nationality=person_data[JsonKeyConfig.information][JsonKeyConfig.p_nationality],
                                                details=person_data[JsonKeyConfig.information][JsonKeyConfig.p_details],
                                                contact_phone=person_data[JsonKeyConfig.information][JsonKeyConfig.p_contact_phone],
                                                photo_paths=person_data[JsonKeyConfig.information][JsonKeyConfig.p_photo_paths],
                                                date=person_data[JsonKeyConfig.information][JsonKeyConfig.p_date],
                                                time=person_data[JsonKeyConfig.information][JsonKeyConfig.p_time]
                                                )
                            self.add_person(new_person)
                    except BaseException:
                        print('Person name:', person_name, 'error. No JSON file')
                        try:
                            shutil.rmtree(f"{DatasetConfig.folder_persons_data}//{person_name}")
                        except BaseException:
                            pass

    def add_person(self, person):
        self.list.append(person)
        self.save_to_file()
        self.check_exist_photo(person)

    def check_exist_photo(self, person):
        found = False
        for i in person.photo_paths:
            if not os.path.exists(i):
                person.photo_paths.remove(i)
                found = True
        if found:
            person.save_json()

    def read_from_file(self):
        person_list = []
        try:
            with open(self.path, 'rb') as input:
                person_list = pickle.load(input).get_list()
        except IOError:
            print("File not accessible")
        except BaseException:
            os.remove(self.path)
        return person_list

    def get_selected(self):  # returns selected model if exists
        selected = None
        for person in self.list:
            if person.selected:
                selected = person
        return selected

    def set_selected(self, name):  # sets model as selected
        for person in self.list:
            person.selected = False
        self.find_first(name).selected = True
        self.save_to_file()

    def is_empty(self):
        return len(self.list) == 0

    def find_first(self, name):  # finds first model that matches name
        found = None
        for person in self.list:
            if person.name == name:
                found = person
        return found

    def check_name_exists(self, name):
        exists = False
        for m in self.list:
            if m.name.lower() == name.lower():
                exists = True
        return exists

    def clear_list(self):
        self.list.clear()
        self.save_to_file()

    def print_list(self):
        for m in self.list:
            print(m.name, m.get_time_created(), m.age)

    def delete_person(self, name):
        person_data_path = self.find_first(name).data_path

        shutil.rmtree(person_data_path)

        name = self.find_first(name).name
        self.list.remove(self.find_first(name))
        self.save_to_file()
        print("Removed:", name)

    def edit_person_name(self, name: str, new_name: str):

        if self.find_first(name) is not None:
            if self.check_name_exists(new_name):
                print("Name already exists")
            else:
                json_path = self.find_first(name).data_path + f"/{DatasetConfig.file_person_json}"
                person_data_dir = self.find_first(name).data_path
                name_index = person_data_dir.rfind(name)
                base_dir = person_data_dir[:name_index]

                self.find_first(name).name = new_name
                with open(json_path, "r") as f:
                    person_data = json.load(f)

                person_data[JsonKeyConfig.information][JsonKeyConfig.p_name] = new_name

                with open(json_path, 'w') as f:
                    json.dump(person_data, f)
                    json.dumps(person_data, indent=4)
                print('RENAME from', DatasetConfig.dir_person_data, 'TO', base_dir + new_name)
                os.rename(DatasetConfig.dir_person_data, base_dir + new_name)
                self.find_first(new_name).data_path = base_dir + new_name
                self.save_to_file()
