import cv2
import base64
import numpy as np
from io import StringIO,BytesIO
from PIL import Image


def get_hog(image, locations=[], winSize=(), winStride=()):
    if len(winSize) == 0:
        winSize = (image.shape[1], image.shape[0])
    if len(winStride) == 0:
        winStride = winSize
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride,
                            cellSize, nbins, derivAperture,
                            winSigma, histogramNormType, L2HysThreshold,
                            gammaCorrection, nlevels)

    padding = (8, 8)
    hist = hog.compute(image, winStride, padding, locations)
    return [x[0] for x in hist]

def readb64(base64_string):
    bytes = base64_string.encode("utf8")
    jpg_bytes = base64.b64decode(bytes)
    f = BytesIO(jpg_bytes)
    pimg = Image.open(f)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2GRAY)

def gen_roi_pos(search_center_pos, w_search_range, h_search_range,  stride ):
    pos_list = []
    w_begin = search_center_pos[0] - w_search_range
    w_end = search_center_pos[0] + w_search_range
    h_begin = search_center_pos[1] - h_search_range
    h_end = search_center_pos[1] + h_search_range
    for w in range(w_begin, w_end, stride):
        for h in range(h_begin, h_end, stride):
            pos_list.append((w,h))
    return  sorted(pos_list, key = lambda x : np.sum(np.square(np.array(x) - np.array(search_center_pos))))