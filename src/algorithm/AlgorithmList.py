# Copyright (C) 2021 Andrii Sonsiadlo

import json
import os
import pickle
import shutil

from UI.person.person import Person
from config import LearningConfig, DatasetConfig, JsonKeyConfig


class AlgorithmList(list):

    def __init__(self):
        super().__init__()

