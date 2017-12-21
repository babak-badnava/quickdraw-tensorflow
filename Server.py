import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from Network import Network
from pre_processing.Config import Config
from ThreadTrainer import ThreadTrainer

from multiprocessing import Queue, Process
from ThreadTester import ThreadTester

class Server():
    def __init__(self):
        self.nb_running_loader = 0
        self.model = Network()

    def train_model(self):
        self.model.train_batch()

    def test_model(self):
        return self.model.test_batch()

    def main(self):
        for e in range(Config.EPOCH):
            e_str = "epoch number : " + str(e)
            self.train_model()
            loss, acc = self.test_model()
            print(e_str + ", batch loss : %.6f, batch accuracy : %.2f" % (loss, acc))

        self.model.finish()


if __name__ == '__main__':
    Server().main()
