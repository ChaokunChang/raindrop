
import os
import cv2
import numpy as np
import torch
from numpy.random import RandomState
from torch.utils.data import Dataset
import h5py
import glob


def precess_train():
    pass


def process_test():
    pass


class RainDataSet(Dataset):
    
    def __init__(self,):
        super(RainDataSet,self).__init__()

    def get_train(self):
        pass

    def get_test(self):
        pass

    def get_dev(self):
        pass

