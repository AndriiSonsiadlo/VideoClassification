# Copyright (C) 2021 Andrii Sonsiadlo

from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox


class RoundButton(Button):
	pass


class CustomRadioButton_static(CheckBox):
	opacity = 0.8
	def on_touch_down(self, *args):
		if self.active:
			return
		super(CustomRadioButton_static, self).on_touch_down(*args)

class CustomRadioButton_toggle(CheckBox):
	opacity = 0.8
	def on_touch_down(self, *args):
		super(CustomRadioButton_toggle, self).on_touch_down(*args)

