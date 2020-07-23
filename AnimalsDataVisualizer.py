import numpy as np
import glob
import pandas
import cv2


maxlength = 20

maxheight = 520

maxwidth = 840

string = input()

pathslist = glob.glob("/Users/21danielw/Downloads/TrainReal/images/" + string + "/*")

video = cv2.VideoWriter(string + "_" + "Video.avi", cv2.VideoWriter_fourcc(*'DIVX'), 15, (840, 520), isColor=True)

for path in pathslist:

    print(path)

    startindex = len("/Users/21danielw/Downloads/TrainReal/images/") + len(string) + len("/") + len(string) + len("_")

    endindex = -4

    index = int(path[startindex:endindex])

    dataframe = pandas.read_csv(open("/Users/21danielw/Downloads/TrainReal/annotations/" + string + ".csv"),
                                header=None)

    tempimagenumpyarray = cv2.imread(path)

    imagenumpyarray = np.zeros((520, 840, 3), dtype="uint8")

    imagenumpyarray[0:tempimagenumpyarray.shape[0], 0:tempimagenumpyarray.shape[1], 0:3] = tempimagenumpyarray

    booleanarray = dataframe.loc[:, 0] == index

    annotationnumpyarray = dataframe.loc[booleanarray, :].to_numpy()

    for row in annotationnumpyarray:

        imagenumpyarray = cv2.rectangle(imagenumpyarray, (row[2], row[3]), (row[2] + row[4], row[3] + row[5]),
                                        (255, 0, 0), 2)

    video.write(imagenumpyarray)

video.release()
