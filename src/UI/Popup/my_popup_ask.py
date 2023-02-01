# Copyright (C) 2021 Andrii Sonsiadlo

from kivy.uix.popup import Popup

# delete  Popup window
from UI.model.model_list import ModelList
from UI.person.person_list import PersonList


class MyPopupAskPerson(Popup):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.list = PersonList()
		self.selected = self.list.get_selected()
		self.title = "Are you sure you want to delete '" + self.selected.name + "'?"

	def yes_pressed(self):
		self.list.delete_person(self.selected.name)
		self.dismiss()

	def no_pressed(self):
		self.dismiss()


class MyPopupAskModel(Popup):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.list = ModelList()
		self.selected = self.list.get_selected()
		self.title = "Are you sure you want to delete '" + self.selected.name + "'?"

	def yes_pressed(self):
		print("yes")
		self.list.delete_model(self.selected.name)
		self.dismiss()

	def no_pressed(self):
		print("no")
		self.dismiss()
