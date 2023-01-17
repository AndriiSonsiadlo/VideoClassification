from dataclasses import dataclass

from objects import ActionElement


@dataclass
class ActionClass:
    class_name: str
    train_list: list[ActionElement]
    test_list: list[ActionElement]

    test_split: float
