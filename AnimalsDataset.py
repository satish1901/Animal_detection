import numpy as np
import torch
import os
import sys
import random
import torch.nn as nn
import torch.utils.data as tdata
import glob
import pandas
import matplotlib.image


class AnimalsDataset(tdata.Dataset):

    def __init__(self):

        self.imagepathsarray = np.array(glob.glob("/Users/21danielw/Downloads/TrainReal/images/*/*.jpg"))

        temparray = self.imagepathsarray.view((str, 1)).reshape(len(self.imagepathsarray), -1)[:, 44:65]

        splicedarray = np.frombuffer(temparray.tostring(), dtype=(str, 65 - 44))

        startarray = np.array(["/Users/21danielw/Downloads/TrainReal/annotations/"])

        endarray = np.array([".csv"])

        self.annotationpathsarray = np.core.defchararray.add(np.core.defchararray.add(startarray, splicedarray),
                                                             endarray)

    def __getitem__(self, index):

        input = matplotlib.image.imread(self.imagepathsarray[index]) / 256

        dataframe = pandas.read_csv(open(self.annotationpathsarray[index]), header=None)

        num = int(self.imagepathsarray[index][88:98])

        booleanarray = dataframe.loc[:, 0] == num

        targetoutput = dataframe.loc[booleanarray, :].to_numpy()

        return input, targetoutput


a = AnimalsDataset()

print(a[3][0].shape)

print(a[3][1].shape)
