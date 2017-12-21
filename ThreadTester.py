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

class ThreadTester(Thread):
    
    def __init__(self, server):
        super(ThreadTester, self).__init__()
        self.server = server
        self.exit_flag = False
        
    def run(self):
        while not self.exit_flag:
            batch_size = 0
            while batch_size <= Config.BATCH_SIZE:
                try:
                    x_, l_ = self.server.testing_q.peek(timeout=0.1)
                except:
                    if self.exit_flag: break
                    continue
                if batch_size == 0:
                    x__ = x_; l__ = l_;
                else:
                    x__ = np.concatenate((x__, x_))
                    l__ = np.append(l__, l_)
                batch_size += x_.shape[0]
            
            loss, accuracy = self.server.test_model(x__, l__)
            
            print("test loss : %.6f, test accuracy : %.2f" % (loss, accuracy))
            sys.stdout.flush()
            
            