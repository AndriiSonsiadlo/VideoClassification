# Copyright (C) 2021 Andrii Sonsiadlo

class MyPhoto:
	size = 0

	def __init__(self, path: str = ''):
		self.index = self.size
		self.__class__.size = self.__class__.size + 1
		self.path = path
