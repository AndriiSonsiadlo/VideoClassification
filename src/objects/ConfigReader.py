import configparser

from objects.Singleton import Singleton


class ConfigReader(metaclass=Singleton):
    def __init__(self):
        self.config_obj = configparser.ConfigParser()
        self.config_obj.read("./config.ini")

    def __call__(self, *args, **kwargs):
        return self.config_obj
