from solution1.objects.Singleton import Singleton


class Logger(metaclass=Singleton):
    def __init__(self):
        #logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
        #logging.basicConfig(level=logging.DEBUG)
        #self.logger = logging.getLogger(__name__)
        pass

