__author__ = 'xiehaoina'
from optparse import OptionParser
import xmltodict
import os
import cv2
import numpy as np
from common import dataset, utils
import time
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pickle
from joblib import dump, load
from functools import reduce
import csv
window_width = 240
window_heigh = 160
scaling_factor = 2



def resize_dataset(dataset, width , height):
    for i in range(0, len(dataset.data)):
        dataset.data[i] = cv2.resize(dataset.data[i], (width,height), interpolation=cv2.INTER_AREA)
    return dataset

def transform_hog_features(dataset):
    for i in range(0, len(dataset.data)):
        dataset.data[i] = utils.get_hog(dataset.data[i])
    return dataset

def train_SVM_mode(X, y):
    clf = svm.SVC(gamma='scale')
    return clf.fit(X, y)

def train_RF_mode(X, y):
    clf = RandomForestClassifier(n_estimators=10)
    return clf.fit(X, y)

def train_model():
    ds = dataset.ImgDataSet(options.input)
    ds = resize_dataset(ds, int(window_width / scaling_factor), int(window_heigh / scaling_factor))
    ds = transform_hog_features(ds)
    clf = train_RF_mode(dataset.data, dataset.target)
    dump(clf, options.model_name)
    return clf

def dump_csv(dataset):
    with open("dist.csv", 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(0, len(dataset.data)):
            img = cv2.resize(dataset.data[i], (80, 64), interpolation=cv2.INTER_AREA)
            # cv2.imwrite(dataset.target[i] + str(i) + ".jpg",img)
            print("=========================")
            print(dataset.target[i])
            print(utils.get_hog(img))
            row = map(lambda x: str(x[0]), utils.get_hog(img)) + [dataset.target[i]]
            spamwriter.writerow(row)
            print("=========================")

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

    parser.add_option(
        "-m", "--model-name",
        action="store",
        dest="model_name",
        type="string",
        default="od.joblib",
        help="specify model_name",
        metavar="INPUT"
    )

    parser.add_option(
        "-t", "--train",
        action="store_true",
        dest="train",
        default=False,
        help="train model"
    )

    (options, args) = parser.parse_args()

    if not os.path.isdir(options.input):
        print("{0} is not a dir".format(options.input))
        exit(0)

    clf = None
    if options.train:
        clf = train_model()
    else:
        clf = load(options.model_name)

    # scores = cross_val_score(clf, dataset.data, dataset.target, cv=5)
    total_img_num = 0
    multi_recognized_img_num = 0
    recognized_img_num = 0
    unrecognized_img_num = 0
    try:

        for file in os.listdir("test"):
            img_path = os.path.join("test",file)
            test_img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            heigh = int(gray_img.shape[0] / scaling_factor)
            width = int(gray_img.shape[1] / scaling_factor)
            image = cv2.resize(gray_img, (width, heigh), interpolation=cv2.INTER_AREA)
            half_heigh = int(heigh / 2)
            half_width = int(width / 2)

            locations = utils.gen_roi_pos((half_width, half_heigh), half_heigh, half_width, 32, 16)

            cross_num , recognized_scaled_imgs, probas , max_prob_index = utils.get_cross_objs(image, clf, locations, int(window_width / scaling_factor), int(window_heigh / scaling_factor))

            #if len(probas) > 0:
            #   max_prob_index,_ = reduce(lambda x, y : x if x[1][0] > y[1][0] else y ,enumerate(probas))

            for i in range(0, len(recognized_scaled_imgs)):
                if i == max_prob_index:
                    cv2.imwrite("cross_"  + file + str(probas[i][0]) + ".jpg", recognized_scaled_imgs[i])
                else:
                    cv2.imwrite("l_cross_" + file + str(probas[i][0]) + ".jpg", recognized_scaled_imgs[i])
            if cross_num == 0:
                unrecognized_img_num += 1
                print("unrecognized file name: {} , unrecognized  files: {} "
                      .format(file, unrecognized_img_num))

            elif cross_num == 1:
                recognized_img_num += 1
                print("recognized file name: {} , recognized  files: {} "
                      .format(file, recognized_img_num))

            elif cross_num >= 2:
                multi_recognized_img_num += 1
                print("multirecognized file name: {} , multirecognized  files: {}, cross numbers: {} "
                      .format(file, multi_recognized_img_num, cross_num))

            total_img_num += 1
    except KeyboardInterrupt:
        print("INFO: stop processing image detection")

    print("total files: {}, unrecognized files: {} , recognized  files: {} , multi_recognized files {}"
          .format( total_img_num, unrecognized_img_num, recognized_img_num, multi_recognized_img_num))



