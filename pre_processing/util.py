import tensorflow as tf
import scipy.misc
import os
from skimage.draw import line_aa
import json
import numpy as np
import matplotlib.pyplot as plt

from Config import Config
from PIL import Image
  
def create_image_from_storks(drawing):
    storks = []
    nb_stork = 0
    
    for stork in drawing:
        points = []            
        nb_stork += 1
        for x, y in zip(stork[0], stork[1]):
            xInd = int(x / 2)
            yInd = int(y / 2)
            if (xInd > 127) : xInd = 127
            if (yInd > 127) : yInd = 127
            points.append((xInd, yInd))
        storks.append(points)
        
    img = np.zeros((Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT), dtype=np.float32)
    for stork in storks:
        for i in range(len(stork) - 1):
            rr, cc, val = line_aa(stork[i][0], stork[i][1], stork[i + 1][0], stork[i + 1][1])
            img[rr, cc] = 1. 
    
#     print(img.shape)
#     img = np.reshape(img, (1, Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT, 1))
#     img = np.reshape(img, (Config.IMAGE_WIDTH * Config.IMAGE_HEIGHT))
#     plt.imshow(img)
#     plt.show()
#     img = Image.fromarray(img, 'RGB')
#     img.show()
#     img.show()
#     print (img)
    return img

def loadfile(filename, dictionary, sampels, labels, nb_sample_per_class):
    idx = 0
    
    f = open(filename) 
    for img in f:
        idx += 1
        img = json.loads(img)
        word = img['word']
        country = img['countrycode']
        recognized = img['recognized']
        if (recognized == 0):
            continue
        
        sample = create_image_from_storks(img['drawing'])
        label = np.array(dictionary[word])
        sampels.append(sample)
        labels.append(label)
         
#         if (idx % 7):
#             self.test_q.put((sample, label))
#         else:
#             self.train_q.put((sample, label))
            
        if (idx > nb_sample_per_class):
            break
    return sampels, labels
        
