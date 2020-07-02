import numpy as np
import torch
import os
import sys
import random
import torch.nn as nn
import torch.utils.data as tdata
import glob
import pandas
import matplotlib.pyplot
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

#matplotlib.pyplot.imshow(a[50000][0])

#matplotlib.pyplot.show()

#print(a[40000][1])

#print(type(a[40000][1]))


maxlength = 20

maxheight = 520

maxwidth = 840


temparray = np.zeros((520, 840))

temparray[0:a[40000][0].shape[0], 0:a[40000][0].shape[1]] = a[40000][0]

print(temparray)


"""
maxheight = 0

maxwidth = 0

for i in range(0, 40661):

    print(str(i) + " " + str(a[i][0].shape[0]) + " " + str(a[i][0].shape[1]))

    if maxheight < a[i][0].shape[0]:

        maxheight = a[i][0].shape[0]

    if maxwidth < a[i][0].shape[1]:

        maxwidth = a[i][0].shape[1]

print(maxheight)

print(maxwidth)
"""


"""
print(a[40000][1])

orderarray = np.lexsort((a[40000][1][:, 5], a[40000][1][:, 4], a[40000][1][:, 3], a[40000][1][:, 2]))

print(orderarray)

print(a[40000][1][orderarray])
"""

"""
temparray = a[40000][1][:, 2:6].reshape(1, -1)

print(temparray)

finalarray = np.zeros((1, 4*20))

finalarray[0:1, 0:temparray.shape[1]] = temparray

print(finalarray)
"""

"""
for i in range(0, 40661):

    print(str(i) + " " + str(a[i][1].shape[0]))

    if maxlength < a[i][1].shape[0]:

        maxlength = a[i][1].shape[0]

print(maxlength)
"""
