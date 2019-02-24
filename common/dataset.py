__author__ = 'xiehaoina'
import tensorflow as tf
import os
import cv2

class ImgDataSet(tf.data.Dataset):
    def __init__(self, input_dir):
        if not os.path.isdir(input_dir):
            self._data = None
            self._target = None
        else:
            self._data = []
            self._target = []
        labels  = os.listdir(input_dir)
        for label in  labels:
            label_dir = os.path.join(input_dir,label)
            for img_file_name in os.listdir(label_dir):
                img_file_path = os.path.join(label_dir,img_file_name)
                self._target.append(labels.index(label))
                img = cv2.imread(img_file_path)
                self._data.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    def _as_variant_tensor(self):
        return None

    def _inputs(self):
        return []

    @property
    def output_types(self):
        return self._output_types

    @property
    def output_shapes(self):
        return self._output_shapes

    @property
    def output_classes(self):
        return self._output_classes

    @property
    def data(self):
        return self._data

    @property
    def target(self):
        return self._target
