__author__ = 'xiehaoina'
from optparse import OptionParser
import xmltodict
import os
import cv2
import numpy as np
from common import dataset
import csv
def get_hog(image):
    winSize = (image.shape[1], image.shape[0])
    blockSize = (8,8)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,
                  cellSize,nbins,derivAperture,
                  winSigma,histogramNormType,L2HysThreshold,
                  gammaCorrection,nlevels)
    winStride = (image.shape[1], image.shape[0])
    padding = (8,8)
    locations = [] # (10, 10)# ((10,20),)
    hist = hog.compute(image,winStride,padding,locations)
    return hist


if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option(
        "-i", "--input",
        action="store",
        dest="input",
        type="string",
        default="input",
        help="specify dir that storing input pictures",
        metavar = "INPUT"
    )

    (options, args) = parser.parse_args()

    if not os.path.isdir(options.input):
        print("{0} is not a dir".format(options.input))
        exit(0)

    dataset = dataset.ImgDataSet(options.input)
    with open("dist.csv",'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(0,len(dataset.data)):
            img = cv2.resize(dataset.data[i], (80,64), interpolation=cv2.INTER_AREA)
            cv2.imwrite(dataset.target[i] + str(i) + ".jpg",img)
            print("=========================")
            print(dataset.target[i])
            print(get_hog(img))
            spamwriter.writerow(get_hog(img).tolist().append(dataset.target[i]))
            print("=========================")


