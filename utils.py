import os
import time
import random
import numpy as np
import cv2
import torch

""" Create a directory. """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)