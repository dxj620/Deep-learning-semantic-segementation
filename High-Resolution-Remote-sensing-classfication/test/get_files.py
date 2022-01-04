import os
import random
import time
import json
import torch
import torchvision
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from torch import nn, optim
from config import config
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset.dataloader import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from timeit import default_timer as timer
from models.model import *
from utils import *
import glob
from IPython import embed
from collections import Counter

def main(root, mode):
    # for test
    all_images=[]
    if mode == "test":
        files = []
        for img in os.listdir(root):
            files.append(root + img)
        files = pd.DataFrame({"filename": files})
        return files
    elif mode != "test":
        # for train and val
        all_data_path, labels = [], []
        # image_folders = list(map(lambda x: root + x, os.listdir(root)))
        image_folders = list(map(lambda x: root+"\\"+x, os.listdir(root)))
        print(image_folders)
        for file in image_folders:
            # print(glob.glob(file + "/*"))
            # one_images = list(chain.from_iterable(list(map(lambda x: glob.glob(x + "/*"), file))))
            all_images.append(glob.glob(file + "/*"))
        print("loading train dataset")
        for files in all_images:
            for file in tqdm(files):
                all_data_path.append(file)
                labels.append(int(file.split("\\")[-2]))
        result = Counter(labels)
        print(result)
        all_files = pd.DataFrame({"filename": all_data_path, "label": labels})
        return all_files
    else:
        print("check the mode please!")

if __name__ == "__main__":
    main("D:\\Competition\\AI_class",'train')
