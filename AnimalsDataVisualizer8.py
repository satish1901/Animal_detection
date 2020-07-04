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
import cv2


maxlength = 20

maxheight = 520

maxwidth = 840

string = input()

pathslist = glob.glob("/Users/21danielw/Downloads/TrainReal/images/" + string + "/*")

"""
print(pathslist)
"""

video = cv2.VideoWriter(string + "_" + "Video.avi", cv2.VideoWriter_fourcc(*'DIVX'), 15, (840, 520), isColor=True)

for path in pathslist:

    print(path)

    """
    print(path)
    """

    startindex = len("/Users/21danielw/Downloads/TrainReal/images/") + len(string) + len("/") + len(string) + len("_")

    endindex = -4

    index = int(path[startindex:endindex])

    dataframe = pandas.read_csv(open("/Users/21danielw/Downloads/TrainReal/annotations/" + string + ".csv"),
                                header=None)

    """
    print(dataframe)

    print(dataframe.shape)
    """

    tempimagenumpyarray = cv2.imread(path)

    """
    print(tempimagenumpyarray.shape)

    print(tempimagenumpyarray.dtype)
    """

    imagenumpyarray = np.zeros((520, 840, 3), dtype="uint8")

    imagenumpyarray[0:tempimagenumpyarray.shape[0], 0:tempimagenumpyarray.shape[1], 0:3] = tempimagenumpyarray

    booleanarray = dataframe.loc[:, 0] == index

    """
    print(booleanarray)
    """

    annotationnumpyarray = dataframe.loc[booleanarray, :].to_numpy()

    """
    print(annotationnumpyarray)
    """

    for row in annotationnumpyarray:

        """
        print(row)
        """

        imagenumpyarray = cv2.rectangle(imagenumpyarray, (row[2], row[3]), (row[2] + row[4], row[3] + row[5]),
                                        (255, 0, 0), 2)

    """
    cv2.imshow("bla", imagenumpyarray)

    cv2.waitKey()
    """

    video.write(imagenumpyarray)

video.release()
