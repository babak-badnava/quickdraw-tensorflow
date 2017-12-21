from multiprocessing import Queue, Value, Lock
import os
from time import sleep

from Config import Config
from  QueueLoaderProcess import  QueueLoaderProcess
import numpy as np
import tensorflow as tf
from tfStartProcess import tfStartProcess
from tfWriteProcess import tfWriteProcess
from tfWriteProcessTest import tfWriteProcessTest
import util


class Server():
    def __init__(self):

        self.cat_filename = os.listdir("../data/simpilified/")
        self.cat_dict = {}
        i = 0
        nb_sample_per_class = 200
        for name in self.cat_filename:
            dot_index = name.index('.')
            self.cat_dict.update({name[:dot_index] : i})
            i += 1
            if (i >= Config.NB_CLASSES):
                break
            
            
        self.training_q = Queue(maxsize=1000)
        self.testing_q = Queue(maxsize=1000)
        
        self.nb_loader_exited = Value('i', 0)
        self.lock = Lock()  
        
        self.start_process = tfStartProcess(self.training_q, self.testing_q, self)
        self.q_loaders = []
        
        self.writer = tfWriteProcess(self.training_q, self.testing_q, self)
        self.writer.start()
        self.test_writer = tfWriteProcessTest(self.training_q, self.testing_q, self)
        self.test_writer.start()
        
    def add_loaders(self):
        i = 0
        for name in self.cat_filename:    
            self.q_loaders.append(QueueLoaderProcess(self.training_q, self.testing_q, "../data/simpilified/" + name, self.cat_dict, self))
            self.q_loaders[-1].start()
            i += 1
            if (i >= Config.NB_CLASSES):
                break
    
    def check_loader_finish(self):
        if (self.training_q.empty() and self.nb_loader_exited.value == len(self.q_loaders)):
            self.writer.exit_flag.value = True
            self.test_writer.exit_flag.value = True
            if (self.nb_loader_exited.value == len(self.q_loaders)):
                self.start_process.exit_flag.value = True
        
    def remove_loader(self, l):
        with self.lock:
            self.nb_loader_exited.value = self.nb_loader_exited.value + 1
                
    def main(self):
        self.start_process.start()

        print("waiting for writer to join")
        self.writer.join()
        print("writer joined")

        print("waiting for test writer to join")
        self.test_writer.join()
        print("test writer joined")
        
        self.start_process.exit_flag = True
        print("waiting for starter to join")
        self.start_process.join()
        print("starter joined")
        
        print("EVERYTHING is done!")    
        return
    
if __name__ == "__main__":
    Server().main()
