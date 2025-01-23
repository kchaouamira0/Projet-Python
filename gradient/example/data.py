#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import pickle # save and load binary files (data, model)
from os.path import dirname, abspath, join

# get the path of this file and get twice the parent directory
pathfile = dirname(dirname(abspath(__file__)))

# function to import and export data from cPickle format
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_obj(obj, name):
    with open('export/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

#import data
data_path = join(pathfile, join('data'))
features = unpickle(join(data_path, 'Sigma_features.pkl'))
labels = unpickle(join(data_path, 'Sigma_labels.pkl'))
labels = labels.reshape(-1,1)