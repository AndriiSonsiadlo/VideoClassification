
# Copyright (C) 2021 Andrii Sonsiadlo

import threading

from kivy import Config
from kivy.app import App
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.gridlayout import GridLayout
from kivy.uix.screenmanager import ScreenManager
from kivy.core.window import Window

from UI.screen_AutomaticMode.automatic_screen import AutomaticMode
from UI.Popup.my_popup_ask import *
from UI.screen_Learning.learningmode_screen import LearningMode
from UI.screen_Learning.editmodel_screen import LearningEdit
from UI.screen_Learning.createmodel_screen import LearningCreate
from UI.screen_AddPerson.addperson_screen import AddPerson
from UI.screen_EditPerson.editperson_screen import EditPerson
from UI.screen_ManualMode.manualmode_screen import ManualMode
from UI.screen_Settings.settings_screen import AppSettings
from UI.screen_Statistics.statistics_screen import Statistics
from UI.screen_ManualMode.selectable_recycleview import *
from UI.screen_Learning.selectable_recycleview_create import *
from UI.drop_button import DropButton
from UI.screen_AutomaticMode.kivy_camera import KivyCamera
from UI.screenstack.screen_stack import ScreenStack

# loading ui files
Builder.load_file("src/assets/kivy/widget_styles.kv")
Builder.load_file("src/assets/kivy/app_ui.kv")
Builder.load_file("src/assets/kivy/automaticmode_screen.kv")
Builder.load_file("src/assets/kivy/addperson_screen.kv")
Builder.load_file("src/assets/kivy/editperson_screen.kv")
Builder.load_file("src/assets/kivy/manualmode_screen.kv")
Builder.load_file("src/assets/kivy/learningmode_screen.kv")
Builder.load_file("src/assets/kivy/createmodel_screen.kv")
Builder.load_file("src/assets/kivy/editmodel_screen.kv")
Builder.load_file("src/assets/kivy/statistics_screen.kv")
Builder.load_file("src/assets/kivy/settings_screen.kv")
Builder.load_file("src/assets/kivy/my_popup.kv")
Builder.load_file("src/assets/kivy/plot_popup.kv")


# Main Screen with navigation bar on top
class Main(GridLayout, threading.Thread):
	manager = ObjectProperty(None)


# manager for changing screens
class WindowManager(ScreenManager):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.stack = ScreenStack()
		self.stack.add_screen("automatic")


# main app class
class AIApp(App):  # Automatic Identification App
	icon = CustomizationConfig.icon
	title = CustomizationConfig.title

	Window.size = (800, 600)
	Window.minimum_width, Window.minimum_height = Window.size
	Window.resizable = False

	def build(self):
		# showing main screen
		return Main()


if __name__ == '__main__':
	AIApp().run()
