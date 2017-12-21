import tensorflow as tf
import os
import numpy as np

from  QueueLoaderProcess import  QueueLoaderProcess
import util
from multiprocessing import Queue, Process, Value
from time import sleep


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class tfWriteProcess(Process):
    
    def __init__(self, training_q, testing_q, server):
        super(tfWriteProcess, self).__init__()
        self.server = server
        self.train_writer = tf.python_io.TFRecordWriter("../data/tf/train/" + "train" + ".tfrecords")

        self.exit_flag = Value('i', False)
        
        self.training_q = training_q
                
    def run(self, *args, **kwargs):
        print("writer begin")
        idx = 0
        while not self.exit_flag.value :
            while not self.training_q.empty():
                try:
                    x_, l_ = self.training_q.get(timeout=0.1)
                except:
                    if self.exit_flag : break
                    continue
                idx += 1
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image_raw': _bytes_feature(x_.tobytes()),
                    'label': _int64_feature(l_)}))
                
                self.train_writer.write(example.SerializeToString())
        
        self.train_writer.close()
        print('writer finished his work')
        return
