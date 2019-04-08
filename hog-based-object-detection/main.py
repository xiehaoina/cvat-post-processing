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

window_width = 240
window_heigh = 160
scaling_size = 2


def get_cross_image(gray_img, sk_model,  x_stride, y_stride , x_size , y_size):
    width = int(gray_img.shape[1] / scaling_size)
    heigh = int(gray_img.shape[0] / scaling_size)
    image = cv2.resize(gray_img, (width, heigh), interpolation=cv2.INTER_AREA)
    cross_num = 0
    recognized_scaled_imgs = []
    probas = []
    for x in range(0, image.shape[1] - x_stride + 1, x_stride):
        for y in range(0, image.shape[0] - y_stride + 1, y_stride):
            sub_img = image[y: y + y_size, x: x + x_size]
            scale_sub_img =  gray_img[scaling_size * y: scaling_size * (y + y_size), scaling_size * x: scaling_size * (x + x_size)]
            if  sub_img.shape == (y_size ,x_size):
                feature = utils.get_hog(sub_img)
                result = sk_model.predict([feature])
                proba = sk_model.predict_proba([feature])
                if result[0] == 0:
                    cross_num += 1
                    recognized_scaled_imgs.append(scale_sub_img)
                    probas.append(proba)
    return cross_num, recognized_scaled_imgs, probas

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
    dataset = resize_dataset(dataset, int(window_width / scaling_size), int(window_heigh / scaling_size))
    dataset = transform_hog_features(dataset)
    clf = train_RF_mode(dataset.data, dataset.target)
    dump(clf,'od.joblib')
    #clf = load('od.joblib')

    scores = cross_val_score(clf, dataset.data, dataset.target, cv=5)

    total_img_num = 0
    multi_recognized_img_num = 0
    recognized_img_num = 0
    unrecognized_img_num = 0
    try:

        for file in os.listdir("test"):
            img_path = os.path.join("test",file)
            test_img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            #small_img = cv2.resize(gray_img, (gray_img.shape[1] / 10,gray_img.shape[0] / 10), interpolation=cv2.INTER_AREA)
            cross_num , recognized_scaled_imgs, probas = get_cross_image(gray_img, clf, 32, 16 , int(window_width / scaling_size), int(window_heigh / scaling_size) )
            for i in range(0, len(recognized_scaled_imgs)):
                cv2.imwrite("cross_"  + file + str(probas[i][0][0]) + ".jpg", recognized_scaled_imgs[i])
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


    '''
    with open("dist.csv",'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(0,len(dataset.data)):
            img = cv2.resize(dataset.data[i], (80,64), interpolation=cv2.INTER_AREA)
            #cv2.imwrite(dataset.target[i] + str(i) + ".jpg",img)
            print("=========================")
            print(dataset.target[i])
            print(get_hog(img))
            row =  map(lambda  x : str(x[0]) ,get_hog(img)) + [dataset.target[i]]
            spamwriter.writerow(row)
            print("=========================")
    '''


