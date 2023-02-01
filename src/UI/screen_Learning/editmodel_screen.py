# Copyright (C) 2021 Andrii Sonsiadlo

from kivy.uix.screenmanager import Screen

from UI.model.model_list import ModelList
from config import LearningConfig


class LearningEdit(Screen):

	def __init__(self, **kw):
		super().__init__(**kw)
		self.model_list = ModelList()
		self.description_deleted = False

	# display model info on screen
	def set_model_data(self, model):
		self.ids.model_name.hint_text = model.name
		self.ids.created_date.text = model.created
		self.ids.author.text = model.author
		if model.algorithm == LearningConfig.algorithm_knn:
			self.ids.threshold_box.height = 30
			self.ids.threshold_box.opacity = 1
		else:
			self.ids.threshold_box.height = 0
			self.ids.threshold_box.opacity = 0

		if model.comment != '':
			self.ids.description.hint_text = model.comment
		else:
			self.ids.description.hint_text = "No description"

	# show info about selected model
	def show_selected(self):
		if not self.model_list.is_empty():
			model = self.model_list.get_selected()
			self.set_model_data(model)

	def save_edited_model(self):
		model = self.model_list.get_selected()
		self.model_list.edit_model(model, self.ids.model_name.text, self.ids.description.text, self.ids.threshold.text)

	def get_default_threshold(self):
		return str(LearningConfig.threshold_default)

	def get_model_threshold(self):
		return str(self.model_list.get_selected().threshold)

	def clear_inputs(self):
		self.description_deleted = False
		self.ids.model_name.text = ''
		self.ids.author.text = ''
		self.ids.description.text = ''
		model = self.model_list.get_selected()
		if model.threshold != LearningConfig.threshold_default:
			self.ids.manual_checkbox.active = True
			self.ids.threshold.text = str("{:}".format(model.threshold))

	def enable_input(self, value):
		if value is True:
			self.ids.threshold.disabled = False
			self.ids.threshold.text = self.get_model_threshold()
		else:
			self.ids.threshold.disabled = True
			self.ids.threshold.text = str(LearningConfig.threshold_default)

	def delete_description(self):
		model = self.model_list.get_selected()
		self.ids.description.text = ''
		self.ids.description.hint_text = 'description deleted'
		self.description_deleted = True
