import os
import re
from collections import defaultdict
from dataclasses import dataclass


@dataclass(init=True, unsafe_hash=True, frozen=True, order=True)
class Action:
    path: str

    @property
    def label(self) -> str:
        re_label = r"[A-Z]{1}[A-Za-z]+[a-z]+"
        label = re.search(re_label, self.filename)
        if label:
            label = label.group()
        else:
            label = "unknown"
            raise Exception("Cannot parse filename to label according to implemented ReGex pattern")
        return label

    @property
    def filename(self) -> str:
        return os.path.split(self.path)[-1]

    @property
    def directory(self) -> str:
        return os.path.join(*os.path.split(self.path)[:-1])

    def __str__(self):
        return self.filename


a1 = Action("C:/v_Act_.avi")
a2 = Action("C:/2_Act_.avi")
a3 = Action("C:/4_Act_.avi")
a4 = Action("C:/v_Bact_.avi")

al = [a1, a2, a3, a4]

# list(filter(lambda data: data), al))
def add_act():
    a = defaultdict
    pass