import os
import pdb
import glob
import json
import numpy as np
import tensorflow as tf

from PIL import Image
import numpy as np
import tensorflow as tf
from pre_processing.Config import Config
from multiprocessing import Process, Queue, Value

class  QueueLoader():
    
    def __init__(self, tfrecords_filename):
        self.filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=Config.EPOCH)
        self.images, self.labels = self.create_reader()
        
    def create_reader(self):
        self.reader = tf.TFRecordReader()
    
        _, serialized_example = self.reader.read(self.filename_queue)
    
        features = tf.parse_single_example(
          serialized_example,
          features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
            })
    
        
        image = tf.decode_raw(features['image_raw'], tf.float32)
#         image = tf.image.decode_jpeg(features['image_raw'])
#         image = Image.frombytes('1', (128, 128), features['image_raw'])
        image = tf.reshape(image, [Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT, 1])
#         im = Image.frombytes('RGB', (128, 128), features['image'])
        label = features['label']
        
        images, labels = tf.train.shuffle_batch([image, label], batch_size=Config.BATCH_SIZE, capacity=Config.QUEUE_SIZE, num_threads=1, min_after_dequeue=10)
        
        return images, labels
        
        
if __name__ == "__main__":
    tfrecords_filename = "./data/tf/test/" + "test" + ".tfrecords"
    
    
    q_loader = QueueLoader(tfrecords_filename)
    # Even when reading in multiple threads, share the filename
    # queue.
    image, annotation = q_loader.images, q_loader.labels
    
    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    
    with tf.Session()  as sess:
        
        sess.run(init_op)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        # Let's read off 3 batches just for example
        for i in range(3):
        
            img, anno = sess.run([image, annotation])
            print(img.shape)
            print(anno)
            print('current batch')
        coord.request_stop()
        coord.join(threads)
    
        
        
        
        
