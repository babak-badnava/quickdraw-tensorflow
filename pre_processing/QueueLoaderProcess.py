import os
import pdb
import glob
import json
import numpy as np
import tensorflow as tf

import numpy as np
import tensorflow as tf
from Config import Config
from multiprocessing import Process, Queue, Value
import util

class  QueueLoaderProcess(Process):
    
    def __init__(self, train_q, test_q, filename, dictonary, server):
        super(QueueLoaderProcess, self).__init__()
        
        self.exit_flag = Value('i', False)        
        
        self.dictionary = dictonary
        self.filename = filename
        self.train_q = train_q
        self.test_q = test_q
        self.server = server
        self.nb_sample_per_class = Config.NB_SAMPLE_PER_CLASS
        
        
    def run(self):
        idx = 0
        f = open(self.filename) 
#         print("process started : " + self.filename)
        
        for img in f:
            idx += 1
            img = json.loads(img)
            word = img['word']
            country = img['countrycode']
            recognized = img['recognized']
            if (recognized == 0):
                continue
            
            sample = util.create_image_from_storks(img['drawing'])
            label = np.array(self.dictionary[word])
            
#             print(str(idx))
            if (idx % 7 == 0):
                self.test_q.put((sample, label))
#                 print("test sample added" + str(idx))
            else:
                self.train_q.put((sample, label))
#                 print("train sample added" + str(idx))
                
            if (self.nb_sample_per_class != None and idx > self.nb_sample_per_class):
                self.exit_flag.value = True
#                 print("process breaked : " + self.filename)
                break
        print("loader of : " + self.filename + " finished his work")
        self.server.remove_loader(self)
        return
        
        
