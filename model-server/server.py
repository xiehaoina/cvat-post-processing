from flask import Flask, request
from flask_restful import Resource, Api
from io import StringIO,BytesIO
from PIL import Image
from joblib import load
import numpy as np
import cv2
import base64
import time

app = Flask(__name__)
api = Api(app)



class predict(Resource):
    w_search_range = 32
    h_search_range = 16
    scaling_factor = 5
    stride = 4
    roi_h_len = int( 320 / scaling_factor )
    roi_w_len = int( 480 / scaling_factor)
    clf = load('od.joblib')
    pre_pos = (128, 96)
    captured = False

    def dynamic_gen_roi_pos(self, image, stride):
        if predict.captured == True:
            predict.w_search_range -= predict.stride
            predict.h_search_range -= predict.stride / 2
        else:
            predict.w_search_range += predict.stride
            predict.h_search_range += predict.stride / 2

        if predict.w_search_range < 2 * predict.stride:
            predict.w_search_range = 2 * predict.stride

        if predict.h_search_range < predict.stride:
            predict.h_search_range = predict.stride

        return self.gen_roi_pos(image, stride)

    def gen_roi_pos(self, image, stride):
        pos_list = []
        print(predict.pre_pos)
        w_begin = predict.pre_pos[0] - predict.w_search_range
        w_end = predict.pre_pos[0] + predict.w_search_range
        h_begin = predict.pre_pos[1] - predict.h_search_range
        h_end = predict.pre_pos[1] + predict.h_search_range
        for w in range(w_begin, w_end, stride):
            for h in range(h_begin, h_end, stride):
                pos_list.append((w,h))
        return  sorted(pos_list, key = lambda x : np.sum(np.square(np.array(x) - np.array(predict.pre_pos))))

    def get_hog(self, image,  locations = [],  winSize = (), winStride = ()):
        if len(winSize) == 0:
            winSize = (image.shape[1], image.shape[0])
        if len(winStride) == 0:
            winStride = winSize
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

        padding = (8,8)
        hist = hog.compute(image,winStride,padding,locations)
        return [x[0] for x in hist]

    def readb64(self, base64_string):
        bytes = base64_string.encode("utf8")
        jpg_bytes = base64.b64decode(bytes)
        f = BytesIO(jpg_bytes)
        pimg = Image.open(f)
        return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2GRAY)

    def chunks(self, l, n):
        # For item i in a range that is a length of l,
        for i in range(0, len(l), n):
            # Create an index range for l of n items:
            yield l[i:i + n]

    def get_rect(self, image_stream):
        result = {"objects":[]}
        print("=======================================")
        print("start read {0}".format(time.time()))
        img_data = self.readb64(image_stream)
        img_h = int(img_data.shape[0] / self.scaling_factor)
        img_w = int(img_data.shape[1] / self.scaling_factor)
        print("start resize {0}".format(time.time()))
        resized_img = cv2.resize(img_data, (img_w, img_h), interpolation=cv2.INTER_AREA)
        print("gen roi {0}".format(time.time()))
        locations = self.gen_roi_pos(resized_img, self.stride)
        print("location len {0}".format(len(locations)))

        hog_features = self.get_hog(resized_img, locations=locations,
                                    winSize=(self.roi_w_len, self.roi_h_len))
        hog_feature_list = list(self.chunks(hog_features, int(len(hog_features) / len(locations))))

        pre_results = predict.clf.predict(hog_feature_list)

        predict.captured = False

        for i in range(0,len(pre_results)):
            if pre_results[i] == 0:
                u_h = locations[i][1]
                d_h = u_h + self.roi_h_len
                l_w = locations[i][0]
                r_w = l_w + self.roi_w_len
                result["objects"].append({
                    "positions":
                        {"cross": [[l_w * self.scaling_factor, u_h * self.scaling_factor]
                            , [r_w * self.scaling_factor, u_h * self.scaling_factor]
                            , [r_w * self.scaling_factor, d_h * self.scaling_factor]
                            , [l_w * self.scaling_factor, d_h * self.scaling_factor]]},
                    "attributes":
                        {"status": "Normal"}
                    }
                )
                predict.pre_pos = (l_w, u_h)
                predict.captured = True
                break

        print(result)
        return result

    def post(self):
        result = {"results": []}
        try:
            data = request.json
            for pic in data["params"]:
                result["results"].append(self.get_rect(pic["data"]))
        except Exception as e:
            print(e)
        return result

api.add_resource(predict,"/")

if __name__ == '__main__':
    #img = cv2.imread("a.jpg")
    #predictor = predict()
    #data = predictor.gen_roi_pos(img, 20)
    #print(data)
    app.run(host='172.16.202.90', port=10001)