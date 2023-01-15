# Copyright (С) 2021 Andrii Sonsiadlo

import csv
import os
import shutil
import threading
import tkinter as tk
from os import path
from tkinter import filedialog

import cv2
import matplotlib.pyplot as plt
import numpy as np
from kivy.core.image import Texture
from kivy.uix.screenmanager import Screen
from matplotlib.ticker import MaxNLocator
from UI.screen_AutomaticMode.kivy_camera import KivyCamera

from UI.Popup.my_popup_person_info import MyPopupPersonInfo
from UI.Popup.my_popup_warn import MyPopupWarn
from UI.Popup.plot_popup import PlotPopup
from UI.get_time import getTime
from UI.model.model_list import ModelList
from UI.person.person import Person
from UI.person.person_list import PersonList
from config import DatasetConfig, StatisticsConfig, LearningConfig, CustomizationConfig

face_scanner_screen = None


class AutomaticMode(Screen):
    loaded_image = None
    model_name = "N/A"
    camera_selected = ""

    LOADED = 0x1
    UNLOADED = 0x0
    photo_status = UNLOADED

    def __init__(self, **kw):
        super().__init__(**kw)
        self.model_list = ModelList()

        try:
            if path.exists(DatasetConfig.path_temp):
                shutil.rmtree(DatasetConfig.path_temp)
            os.mkdir(DatasetConfig.path_temp)

            if not os.path.exists(StatisticsConfig.path_file_stats):
                open(StatisticsConfig.path_file_stats, 'tw', encoding='utf-8').close()
        except Exception:
            pass

    def refresh(self):
        self.ids.model_name.values = self.get_values_model()

    def camera_on_off(self):
        self.toggle_camera()

    def toggle_camera(self):
        if self.photo_status:
            self.clear_photo()
        self.person_list.read_from_file()
        self.disable_button(self.ids.identification_btn)
        KivyCamera.on_off(self.ids.camera, self, self.model_list.get_selected(), self.camera_selected)

    # get names of the model dropdown menu
    def get_values_model(self):
        values = list(self.model_list.keys())

        if not values:
            values.append("N/A")
            self.ids.model_name.text = values[0]
        else:
            model = self.model_list.get_selected()
            if not model:  # show last model if none has been selected
                model = list(self.model_list.values())[-1]
                self.model_list.set_selected(model.name)
            self.set_model_data(model)
        return values

    def set_text_camera_spinner(self):
        self.camera_selected = CustomizationConfig.camera_values[0]
        return self.camera_selected

    def on_spinner_camera_select(self, camera):
        self.camera_selected = camera

    def get_values_camera(self):
        if CustomizationConfig.camera_values:
            return CustomizationConfig.camera_values
        else:
            return ["N/A"]

    def on_spinner_model_select(self, name):
        model = self.model_list.get(name)
        if model is not None:
            self.model_list.set_selected(model.name)
            self.set_model_data(model)

    def set_model_data(self, model):
        self.ids.model_name.text = model.name
        print("Loaded model:", model.name, model.created, model.author, model.comment, model.clf_path)

    def model_load_list(self):
        self.model_list = ModelList()

    def person_load_list(self):
        self.person_list = PersonList()
        self.person_list.update_person_list()

    def set_text_number_in_base(self):

        # self.person_list.update_person_list()
        # self.person_list.read_from_file()
        return str(len(PersonList().get_list()))

    def switch_on_person(self, name):
        if (KivyCamera.get_status_camera(self.ids.camera)):
            KivyCamera.clock_unshedule(self.ids.camera)

        person = self.person_list.find_first(name)
        if person is not None:
            self.show_popup_person_info(person=person)
        else:
            self.show_popup_warm(title=f"{name} not found in database")

    def show_popup_warm(self, title):
        popupWindow = MyPopupWarn(text=title)
        popupWindow.open()

    def show_popup_person_info(self, person: Person):
        popupWindow = MyPopupPersonInfo(person=person)
        popupWindow.open()

    def disable_button(self, button):
        if (button == self.ids.identification_btn):
            button.text = "N/A"
            self.disable_button(self.ids.its_ok_btn)
            self.disable_button(self.ids.its_nok_btn)

        button.disabled = True
        button.opacity = .5

    def enable_button(self, button, name=''):
        if button == self.ids.identification_btn:
            button.text = name
            self.enable_button(button=self.ids.its_ok_btn)
            self.enable_button(button=self.ids.its_nok_btn)
            self.its_add_one()
        button.disabled = False
        button.opacity = 1

    def read_plot(self):
        data_csv = []
        if os.path.exists(StatisticsConfig.path_file_stats):
            with open(StatisticsConfig.path_file_stats, 'r') as csvfile:
                file_rd = csv.reader(csvfile, delimiter=',')
                for (i, row) in enumerate(file_rd):
                    data_csv.append(row)
        else:
            # self.clear_stats()
            return StatisticsConfig.path_plt_facescreen_stats

        try:
            len1 = len(data_csv[0])
        except BaseException:
            self.clear_stats()
            return StatisticsConfig.path_plt_facescreen_stats

        if (len1):
            current_hour = int(getTime("hour"))
            current_day = int(getTime("day"))
            current_month = int(getTime("month"))

            range_hours = []
            range_day = []

            if (current_hour - 11) < 0:
                first_hour = 24 - (11 - current_hour)
                range_day.append(current_day - 1)
                range_day.append(current_day)
            else:
                first_hour = current_hour - 11
                range_day.append(current_day)

            #			with open(path_file_stats, "a", newline='') as gen_file:
            #				writing = csv.writer(gen_file, delimiter=',')
            #				data = [current_hour, current_day, current_month, 0, 0, 0]
            #				writing.writerow(data)

            for hour in range(12):
                if (first_hour + hour > current_hour) and (first_hour + hour < 24):
                    range_hours.append(first_hour + hour)
                elif first_hour + hour >= 24:
                    range_hours.append((first_hour + hour) - 24)
                else:
                    range_hours.append(first_hour + hour)

            x = []
            ok_y = []
            nok_y = []
            nnok_y = []

            for hour in range_hours:
                ok = 0
                nok = 0
                nnok = 0

                for element2 in data_csv:
                    if int(element2[0]) == hour and int(element2[1]) in range_day:
                        nnok += int(element2[3])
                        ok += int(element2[4])
                        nok += int(element2[5])
                x.append(hour)
                if nnok - nok - ok < 0:
                    nnok_y.append(0)
                else:
                    nnok_y.append(nnok - nok - ok)
                nok_y.append(nok)
                ok_y.append(ok)

            # for element1 in data_csv:
            # 	hour = int(element1[0])
            # 	day = int(element1[1])
            # 	month = int(element1[2])
            #
            # 	if (not hour in black_list) and (hour in range_hours) and (day in range_day) and (month == current_month):
            # 		black_list.append(hour)
            # 		ok = 0
            # 		nok = 0
            # 		nnok = 0
            #
            # 		for element2 in data_csv:
            # 			if int(element2[0]) == int(hour):
            # 				nnok += int(element2[3])
            # 				ok += int(element2[4])
            # 				nok += int(element2[5])
            # 		x.append(hour)
            # 		if nnok-nok-ok < 0:
            # 			nnok_y.append(0)
            # 		else:
            # 			nnok_y.append(nnok-nok-ok)
            # 		nok_y.append(nok)
            # 		ok_y.append(ok)

            series1 = np.array(ok_y)
            series2 = np.array(nok_y)
            series3 = np.array(nnok_y)

            index = np.arange(len(x))
            plt.title('Result of identification (per hour)')
            plt.ylabel('Count of identificated')
            plt.xlabel('Hour')

            plt.bar(index, series1, color="g")
            plt.bar(index, series2, color="r", bottom=series1)
            plt.bar(index, series3, color="b", bottom=(series2 + series1))
            # plt.tight_layout()
            plt.xticks(index, x)
            ax = plt.gca()
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(axis='y')
            plt.savefig(StatisticsConfig.path_plt_facescreen_stats)
            plt.clf()

            self.update_right_plot(StatisticsConfig.path_plt_facescreen_stats)
        else:
            self.clear_stats()

            self.update_right_plot(StatisticsConfig.path_plt_facescreen_stats)
        return StatisticsConfig.path_plt_facescreen_stats

    def clear_stats(self):
        current_hour = int(getTime("hour"))
        current_day = int(getTime("day"))
        current_month = int(getTime("month"))

        range_hours = []

        if (current_hour - 11) < 0:
            first_hour = 24 - current_hour - 11

        else:
            first_hour = current_hour - 11

        for hour in range(12):
            range_hours.append(first_hour + hour)

        with open(StatisticsConfig.path_file_stats, "w", newline='') as gen_file:
            writing = csv.writer(gen_file, delimiter=',')
            for hour in range_hours:
                if hour > current_hour:
                    data = [hour, current_day - 1, current_month, 0, 0, 0]
                    writing.writerow(data)
                else:
                    data = [hour, current_day, current_month, 0, 0, 0]
                    writing.writerow(data)

        plt.title('Result of identification (per hour)')
        plt.ylabel('Count of identificated')
        plt.xlabel('Hour')
        plt.xticks(np.arange(len(range_hours)), range_hours)
        ax = plt.gca()
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis='y')
        plt.savefig(StatisticsConfig.path_plt_facescreen_stats)
        plt.clf()
        self.read_plot()

    def popup_photo(self):
        if (os.path.exists(StatisticsConfig.path_plt_facescreen_stats)):
            try:
                popup_window = PlotPopup(self.ids.plot.source)
                popup_window.open()
                pass
            except BaseException:
                pass

    def update_right_plot(self, source):
        self.ids.plot.source = source
        self.ids.plot.reload()

    def its_ok(self):
        self.disable_button(button=self.ids.its_ok_btn)
        self.disable_button(button=self.ids.its_nok_btn)

        current_hour = int(getTime("hour"))
        current_day = int(getTime("day"))
        current_month = int(getTime("month"))
        data = [current_hour, current_day, current_month, 0, 1, 0]

        with open(StatisticsConfig.path_file_stats, "a", newline='') as gen_file:
            writing = csv.writer(gen_file, delimiter=',')
            writing.writerow(data)
        self.read_plot()

    def its_nok(self):
        self.disable_button(button=self.ids.its_ok_btn)
        self.disable_button(button=self.ids.its_nok_btn)

        current_hour = int(getTime("hour"))
        current_day = int(getTime("day"))
        current_month = int(getTime("month"))

        data = [current_hour, current_day, current_month, 0, 0, 1]

        with open(StatisticsConfig.path_file_stats, "a", newline='') as gen_file:
            writing = csv.writer(gen_file, delimiter=',')
            writing.writerow(data)
        self.read_plot()

    def its_add_one(self):
        current_hour = int(getTime("hour"))
        current_day = int(getTime("day"))
        current_month = int(getTime("month"))

        data = [current_hour, current_day, current_month, 1, 0, 0]

        with open(StatisticsConfig.path_file_stats, "a", newline='') as gen_file:
            writing = csv.writer(gen_file, delimiter=',')
            writing.writerow(data)
        self.read_plot()
