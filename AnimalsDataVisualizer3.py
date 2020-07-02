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

        tempinputarray = matplotlib.image.imread(self.imagepathsarray[index]) / 256

        input = np.zeros((520, 840))

        input[0:tempinputarray.shape[0], 0:tempinputarray.shape[1]] = tempinputarray

        dataframe = pandas.read_csv(open(self.annotationpathsarray[index]), header=None)

        num = int(self.imagepathsarray[index][88:98])

        booleanarray = dataframe.loc[:, 0] == num

        numpyarray = dataframe.loc[booleanarray, :].to_numpy()

        orderarray = np.lexsort((numpyarray[:, 5], numpyarray[:, 4], numpyarray[:, 3], numpyarray[:, 2]))

        sortednumpyarray = numpyarray[orderarray]

        temparray = sortednumpyarray[:, 2:6].reshape(-1)

        targetoutput = np.zeros(4*20)

        targetoutput[0:temparray.shape[0]] = temparray

        return input, targetoutput

    def __len__(self):

        return self.imagepathsarray.shape[0]


def displayimage(index):

    dataset = AnimalsDataset()

    figure = matplotlib.pyplot.figure()

    subplot = figure.add_subplot(111)

    subplot.imshow(dataset[index][0], cmap="gray")

    for i in range(0, 20):

        rectangularpatch = matplotlib.patches.Rectangle((dataset[index][1][i * 4], dataset[index][1][i * 4 + 1]),
                                                        dataset[index][1][i * 4 + 2], dataset[index][1][i * 4 + 3],
                                                        edgecolor="r", facecolor="none")

        subplot.add_patch(rectangularpatch)

    matplotlib.pyplot.show()


displayimage(30000)
