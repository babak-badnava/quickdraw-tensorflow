import tensorflow as tf
import os
import numpy as np

from  QueueLoaderProcess import  QueueLoaderProcess
import util
from multiprocessing import Queue, Process, Value
from time import sleep
from tfWriteProcess import tfWriteProcess

class tfStartProcess(Process):
    def __init__(self, training_q, testing_q, server):
        super(tfStartProcess, self).__init__()
        
        self.server = server
        self.exit_flag = Value('i', False)        
    
    def run(self, *args, **kwargs):
        self.server.add_loaders()

        while(not self.exit_flag.value):
            self.server.check_loader_finish()
            sleep(0.2)
            
        print("starter finished his work")
        return


