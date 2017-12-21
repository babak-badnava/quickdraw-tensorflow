import os
import pdb
import glob

import sys
import numpy as np
import tensorflow as tf

import numpy as np
import tensorflow as tf

import time
from pre_processing.Config import Config
from multiprocessing import Process, Queue
from threading import Thread

class ThreadTrainer(Thread):
    
    def __init__(self, server):
        super(ThreadTrainer, self).__init__()
        self.server = server
        self.exit_flag = False
        self.batch_num = 0
        
    def run(self):
        while not self.exit_flag:
            batch_size = 0
            while batch_size <= Config.BATCH_SIZE:
                try:
                    x_, l_ = self.server.training_q.get(timeout=0.1)
                except:
                    if self.exit_flag: break
                    continue
                if batch_size == 0:
                    x__ = x_; l__ = l_;
                else:
                    x__ = np.concatenate((x__, x_))
                    l__ = np.append(l__, l_)
                batch_size += x_.shape[0]
            
            self.server.train_model(x__, l__)
            self.batch_num += 1
            if (self.batch_num % 1 == 0):
                loss, accuracy = self.server.test_model(x__, l__)
                print("test loss : %.6f, test accuracy : %.2f" % (loss, accuracy))
                sys.stdout.flush()
            
            
            
