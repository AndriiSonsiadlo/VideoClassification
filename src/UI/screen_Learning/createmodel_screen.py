# Copyright (C) 2021 Andrii Sonsiadlo
import os
import re
from tkinter import filedialog
import shutil
from dataclasses import dataclass
import tkinter as tk
from kivy.clock import mainthread
from kivy.uix.screenmanager import Screen

from UI.Popup.my_popup_warn import MyPopupWarn
from UI.Popup.plot_popup import PlotPopup
from UI.model.model import Model
from UI.model.model_list import ModelList
from UI.person.person_list import PersonList
from algorithm.InceptionV3 import InceptionV3
from config import CustomizationConfig, LearningConfig


@dataclass
class LearningStates:
    WAITING = 0x0
    READY = 0x01
    IN_PROGRESS = 0x10
    COMPLETED = 0x11


class LearningCreate(Screen):
    algorithms = {
        InceptionV3.name: InceptionV3,
    }

    learning_state = LearningStates.WAITING

    def __init__(self, **kw):
        super().__init__(**kw)
        self.selected_video_paths = []
        # self.action_list = ActionList()

    @mainthread
    def clear_inputs(self):
        if self.learning_state == LearningStates.WAITING:
            self.selected_video_paths = []

            self.enable_epoch_input(False, self.ids.epochs_number)

            # clear inputs
            self.ids.model_name.text = ''
            self.ids.author.text = ''
            self.ids.comment.text = ''

            self.on_spinner_select_algorithm(self.ids.spinner_algorithm.text)
            self.ids.features_number.text = ''
            self.ids.max_seq_length.text = ''
            self.ids.epochs_number.text = ''
            self.ids.batch_size.text = ''
            self.ids.max_video_frames.text = ''
            self.ids.frame_size_w.text = ''
            self.ids.frame_size_h.text = ''

            if len(self.selected_video_paths) > 0:
                self.ids.import_videos_loaded.text = str(len(self.selected_video_paths)) + " loaded"
                self.ids.import_videos_loaded.opacity = 1
                self.ids.dir_icon.opacity = 0
                self.set_train_model_btn(LearningStates.READY)
            else:
                self.ids.import_videos_loaded.opacity = 0
                self.ids.dir_icon.opacity = 1
                self.set_train_model_btn(LearningStates.WAITING)

    def set_train_model_btn(self, state: int):
        if state == LearningStates.WAITING:
            self.ids.import_videos_btn.disabled = False
            self.ids.begin_learning_btn.text = CustomizationConfig.text_train_model
            self.ids.begin_learning_btn.disabled = True
            self.ids.begin_learning_btn.opacity = 0
            self.ids.learning_results.opacity = 0
        elif state == LearningStates.READY:
            self.ids.import_videos_btn.disabled = False
            self.ids.begin_learning_btn.disabled = False
            self.ids.begin_learning_btn.opacity = 1
            self.ids.begin_learning_btn.text = CustomizationConfig.text_train_model
            self.ids.begin_learning_btn.color = CustomizationConfig.normal_text_color
        elif state == LearningStates.IN_PROGRESS:
            self.ids.import_videos_btn.disabled = True
            self.ids.begin_learning_btn.text = CustomizationConfig.text_learning
            self.ids.begin_learning_btn.color = CustomizationConfig.normal_text_color
        elif state == LearningStates.COMPLETED:
            self.ids.begin_learning_btn.text = CustomizationConfig.text_completed
            self.show_results(learning_time=self.new_model.learning_time, threshold=self.new_model.threshold,
                              accuracy=self.new_model.accuracy)
            self.ids.begin_learning_btn.disabled = True
            self.ids.begin_learning_btn.opacity = .5

    @mainthread
    def load_videos(self):
        root = tk.Tk()
        root.withdraw()
        video_paths = filedialog.askopenfilenames(filetypes=[("Video files", ".avi .mp4")])
        self.selected_video_paths = list(
            filter(lambda paths: (path if not os.path.split(path)[-1].startswith("._") else ... for path in paths),
                   video_paths))
        num_loaded = len(self.selected_video_paths)

        if num_loaded > 0:
            self.ids.import_videos_loaded.text = str(num_loaded) + " loaded"
            self.ids.import_videos_loaded.opacity = 1
            self.ids.dir_icon.opacity = 0
        self.get_root_window().raise_window()

        self.set_train_model_btn(LearningStates.READY)

    def begin_learning(self):
        person_list = PersonList()
        model_list = ModelList()

        if self.ids.begin_learning_btn.text == CustomizationConfig.text_learning:
            if len(person_list.get_list_with_photo()):
                self.new_model.name = self.ids.model_name.text
                self.new_model.author = self.ids.author.text
                self.new_model.comment = self.ids.comment.text

                if self.new_model.name == "":
                    self.new_model.name = CustomizationConfig.text_unnamed
                    self.ids.model_name.text = self.new_model.name
                if self.new_model.author == "":
                    self.new_model.author = CustomizationConfig.text_unknown
                    self.ids.author.text = self.new_model.author
                if model_list.check_name_exists(self.new_model.name):
                    print("File", self.new_model.name, "already exists")
                    repeated = 1
                    while model_list.check_name_exists(self.new_model.name + "(" + str(repeated) + ")"):
                        repeated += 1
                    self.new_model.name += "(" + str(repeated) + ")"
                    self.ids.model_name.text = self.new_model.name

                self.new_model.path_model_data = os.path.join(LearningConfig.folder_models_data, self.new_model.name)
                if not os.path.isdir(LearningConfig.folder_models_data):
                    os.mkdir(LearningConfig.folder_models_data)
                if not os.path.isdir(self.new_model.path_model_data):
                    os.mkdir(self.new_model.path_model_data)

                if self.ids.neighbor_checkbox.active == False:
                    n_neighbor = None
                else:
                    try:
                        n_neighbor = int(self.ids.create_neighbor_num.text)
                    except BaseException:
                        n_neighbor = None

                learned_succes, title_warn = self.new_model.begin_learning(algorithm=self.algorithm_selected,
                                                                           n_neighbor=n_neighbor,
                                                                           weight=self.weight_selected,
                                                                           gamma=self.gamma_selected)
                if (learned_succes):
                    print(self.new_model.learning_time)

                    model_list.add_model(self.new_model)
                    model_list.set_selected(model_list.get_list()[-1].name)
                    self.set_train_model_btn(LearningStates.COMPLETED)
                    self.new_model = Model()
                else:
                    self.show_popup_warm(title=title_warn)
                    self.ids.begin_learning_btn.text = CustomizationConfig.text_train_model
                    self.ids.begin_learning_btn.color = CustomizationConfig.normal_text_color
                    try:
                        shutil.rmtree(f"{self.new_model.path_model_data}")
                    except BaseException:
                        pass
            else:
                self.show_popup_warm(title="Not found persons with a photo")
        self.isLearning = False

    @mainthread
    def show_results(self, learning_time, threshold, accuracy):
        threshold = str(round(threshold, 5))
        if self.algorithm_selected == LearningConfig.algorithm_knn:
            self.ids.learning_results.text = f"Learning time: {learning_time} s, accuracy: {accuracy}, default threshold: {threshold}"
        else:
            self.ids.learning_results.text = f"Learning time: {learning_time} s, accuracy: {accuracy}"
        self.ids.learning_results.opacity = 1

    @mainthread
    def show_plot(self, data_path):
        plot_path = os.path.join(data_path, 'plot.png')
        popupWindow = PlotPopup(plot_path)
        popupWindow.open()

    def show_popup_warm(self, title):
        popupWindow = MyPopupWarn(text=title)
        popupWindow.open()

    def enable_epoch_input(self, active, inputfield):
        if active:
            inputfield.disabled = False
            inputfield.text = ''
            inputfield.hint_text = 'auto'
        else:
            inputfield.disabled = True
            inputfield.text = 'auto'

    def on_spinner_select_algorithm(self, algorithm):
        if algorithm == InceptionV3.name:
            self.ids.inceptionv3_settings_box.opacity = 1
            self.set_text_weights_spinner()
            # self.ids.svm_algorithm_box.opacity = 0
        elif algorithm == "":
            # self.ids.svm_algorithm_box.opacity = 1
            self.ids.inceptionv3_settings_box.opacity = 0

    def get_values_algorithm(self):
        return list(self.algorithms.keys()) if self.algorithms else []

    def set_text_algorithm_spinner(self):
        return self.get_values_algorithm()[0]

    def get_values_weights(self):
        return LearningConfig.weights_values if LearningConfig.weights_values else []

    def set_text_weights_spinner(self):
        return self.get_values_weights()[0]
