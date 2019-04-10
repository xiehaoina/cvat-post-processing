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

def gen_roi_pos(search_center_pos, w_search_range, h_search_range,  w_stride, h_stride , sorted = False):
    # initialize search range
    pos_list = []
    w_begin = search_center_pos[0] - w_search_range
    w_end = search_center_pos[0] + w_search_range
    h_begin = search_center_pos[1] - h_search_range
    h_end = search_center_pos[1] + h_search_range
    # Limit range
    w_begin = 0 if w_begin < 0 else w_begin
    h_begin = 0 if h_begin < 0 else h_begin
    # Generate positions
    for w in range(w_begin, w_end, w_stride):
        for h in range(h_begin, h_end, h_stride):
            pos_list.append((w,h))
    # Sort positions
    if sorted:
        return  sorted(pos_list, key = lambda x : np.sum(np.square(np.array(x) - np.array(search_center_pos))))
    else:
        return pos_list


def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]


def get_cross_objs(image, classifer, locations, w_winsize, h_winsize):
    hog_features = get_hog(image, locations=locations, winSize=(w_winsize, h_winsize))

    hog_feature_list = list(chunks(hog_features, int(len(hog_features) / len(locations))))

    predict_results = classifer.predict(hog_feature_list)
    predict_probs = classifer.predict_proba(hog_feature_list)
    recognized_scaled_imgs = []
    probas = []
    cross_num = 0
    max_prob_index = 0
    for i in range(0, len(predict_results)):
        if predict_results[i] == 0:
            cross_num += 1
            probas.append(predict_probs[i])

            max_prob_index = max_prob_index if probas[max_prob_index][0] > predict_probs[i][0] else len(probas) - 1

            w = locations[i][1]
            h = locations[i][1]
            recognized_scaled_imgs.append(image[h: h + h_winsize, w: (w + w_winsize)])

    return cross_num, recognized_scaled_imgs, probas, max_prob_index