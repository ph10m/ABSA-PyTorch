import logging
from sys import stdout
from time import localtime, strftime


class Logger:
    def __init__(self, name, data):
        if not name:
            name = 'missing-log-name'

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        folder = 'logs'
        current_time = strftime("%y%m%d-%H%M", localtime())
        _file = '{}/{}-{}-{}.log'.format(folder, name, data, current_time)
        self.logger.addHandler(logging.FileHandler(_file))
        self.logger.addHandler(logging.StreamHandler(stdout))

    def log(self, *args):
        self.logger.info(*args)
