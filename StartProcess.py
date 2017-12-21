import os
import pdb
import glob

import numpy as np
import tensorflow as tf

import numpy as np
import tensorflow as tf
from Config import Config

import time
from pre_processing.Config import Config
from multiprocessing import Process, Queue

class StartProcess(Process):
    
    def __init__(self, server):
        super(StartProcess, self).__init__()
        self.server = server
        self.cat_filename = os.listdir("./data/simpilified/")
        
        print(self.cat_filename)
        self.cat_dict = {}
        i = 0
        for name in self.cat_filename:
            dot_index = name.index('.')
            self.cat_dict.update({name[:dot_index] : i})
            i += 1
        
    def run(self, *args, **kwargs):
        for i in range(Config.NB_TRAINER):
            self.server.add_trainer()
#         self.server.add_tester()
            
        for filename in self.cat_filename:
            while(self.server.nb_running_loader > Config.RUNNING_LOADER_COUNT):
                time.sleep(1)
            self.server.add_loader("./data/simpilified/" + filename, self.cat_dict)
            
            
            
            