# Copyright (C) 2021 Andrii Sonsiadlo

import re
from kivy.uix.screenmanager import Screen
from UI.Popup.my_popup_ask import MyPopupAskModel
from UI.model.model_list import ModelList
from UI.person.person_list import PersonList


class LearningMode(Screen):
    model_name = "N/A"
    created = "N/A"
    author = "N/A"
    comment = "N/A"
    model_list: ModelList

    def __init__(self, **kw):
        super().__init__(**kw)
        self.model_list = ModelList()

    def refresh(self):
        self.model_list.read_from_file()
        self.ids.model_name.values = self.get_values_model()

    # display model info on screen
    def set_model_data(self, name):
        model = self.model_list.get(name)
        self.ids.model_name.text = model.name
        self.ids.created_date.text = model.created
        self.ids.author.text = model.author
        self.ids.comment.text = model.comment
        if model.comment != '':
            self.ids.comment.opacity = 1
        else:
            self.ids.comment.opacity = 0

        self.ids.algorithm_text.text = model.algorithm

        # self.ids.neighbor_box.height = 30
        # self.ids.neighbor_box.opacity = 1
        # self.ids.threshold_box.height = 30
        # self.ids.threshold_box.opacity = 1
        # self.ids.weight_box.height = 30
        # self.ids.weight_box.opacity = 1
        #
        # self.ids.gamma_box.height = 0
        # self.ids.gamma_box.opacity = 0
        #
        # self.ids.threshold.text = str(model.threshold)
        # self.ids.num_neighbors.text = str(model.n_neighbor)
        # self.ids.weight.text = str(model.weight)
        #
        #
        # self.ids.gamma_box.height = 30
        # self.ids.gamma_box.opacity = 1
        # self.ids.gamma.text = str(model.gamma)
        #
        # self.ids.neighbor_box.height = 0
        # self.ids.neighbor_box.opacity = 0
        # self.ids.threshold_box.height = 0
        # self.ids.threshold_box.opacity = 0
        # self.ids.weight_box.height = 0
        # self.ids.weight_box.opacity = 0

        self.ids.learning_time.text = str(model.learning_time)
        self.ids.accuracy.text = str(model.accuracy)
        # self.ids.num_trained.text = str(model.count_train_Y)
        # self.ids.num_tested.text = str(model.count_test_Y)
        # self.ids.num_all.text = str(len(PersonList().get_list()))

        print("Loaded model:", model.name, model.created, model.author, model.comment, model.clf_path)




    # clear on screen model info
    def clear_model_data(self):
        self.ids.model_name.text = "N/A"
        self.ids.created_date.text = "N/A"
        self.ids.author.text = "N/A"
        self.ids.comment.text = "N/A"
        self.ids.algorithm_text.text = "N/A"
        self.ids.learning_time.text = "N/A"
        self.ids.accuracy.text = "N/A"
        self.ids.num_trained.text = "N/A"
        self.ids.num_tested.text = "N/A"
        self.ids.num_all.text = "N/A"

        self.ids.num_neighbors.text = "N/A"
        self.ids.weight.text = "N/A"
        self.ids.threshold.text = "N/A"

        self.ids.gamma_box.height = 0
        self.ids.gamma_box.opacity = 0
        self.ids.neighbor_box.height = 0
        self.ids.neighbor_box.opacity = 0
        self.ids.weight_box.height = 0
        self.ids.weight_box.opacity = 0
        self.ids.threshold_box.height = 0
        self.ids.threshold_box.opacity = 0

        self.ids.comment.opacity = 0
        self.ids.train_dataset.opacity = 0

    # get names of the model dropdown menu
    def get_values_model(self):
        values = list(self.model_list.keys())
        print(values)
        if not values:
            values.append("N/A")
            self.ids.model_name.text = values[0]
            self.clear_model_data()
            self.disable_button(self.ids.edit_btn)
            self.disable_button(self.ids.delete_btn)
            self.model_list.selected_model = None
        else:
            model = self.model_list.get_selected()
            if not model:  # show last model if none has been selected
                model = list(self.model_list.values())[-1]
            self.model_list.set_selected(model.name)
            self.set_model_data(model.name)
            self.enable_button(self.ids.edit_btn)
            self.enable_button(self.ids.delete_btn)
        return values

    def on_spinner_model_select(self, name):
        model = self.model_list.get(name)
        self.model_list.set_selected(model.name)
        self.set_model_data(model.name)
    def disable_button(self, button):
        button.disabled = True
        button.opacity = .5

    def enable_button(self, button):
        button.disabled = False
        button.opacity = 1

    def show_popup(self):
        selected = self.model_list.get_selected()
        if selected is not None:
            print(selected)
            popupWindow = MyPopupAskModel()
            popupWindow.bind(on_dismiss=self.popup_refresh)
            popupWindow.open()

    def popup_refresh(self, instance):  # update screen after pressing delete
        self.refresh()
