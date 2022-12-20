from abc import ABC, ABCMeta, abstractmethod


class Algorithm(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def validate(self, *args, **kwargs):
        raise NotImplementedError
