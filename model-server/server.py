from flask import Flask, request
from flask_restful import Resource, Api
from joblib import load
import cv2
import time
import sys

sys.path.append("..")
from common import utils

app = Flask(__name__)
api = Api(app)

class predict(Resource):
    w_search_range = 32
    h_search_range = 16
    scaling_factor = 2
    stride = 2
    roi_h_len = int( 320 / scaling_factor )
    roi_w_len = int( 480 / scaling_factor)
    clf = load('od.joblib')
    pre_pos = (128, 96)
    captured = False

    def dynamic_gen_roi_pos(self, image, stride):
        if predict.captured == True:
            predict.w_search_range -= predict.stride
            predict.h_search_range -= int(predict.stride / 2)
        else:
            predict.w_search_range += predict.stride
            predict.h_search_range += int(predict.stride / 2)

        if predict.w_search_range < 2 * predict.stride:
            predict.w_search_range = 2 * predict.stride

        if predict.h_search_range < predict.stride:
            predict.h_search_range = predict.stride

        return utils.gen_roi_pos(predict.pre_pos, predict.w_search_range, predict.h_search_range, stride)




    def get_rect(self, image_stream):
        result = {"objects":[]}
        print("=======================================")
        print("start read {0}".format(time.time()))
        img_data = utils.readb64(image_stream)
        img_h = int(img_data.shape[0] / self.scaling_factor)
        img_w = int(img_data.shape[1] / self.scaling_factor)
        print("start resize {0}".format(time.time()))
        resized_img = cv2.resize(img_data, (img_w, img_h), interpolation=cv2.INTER_AREA)
        print("gen roi {0}".format(time.time()))
        locations = self.dynamic_gen_roi_pos(resized_img, self.stride)
        print("location len {0}".format(len(locations)))

        hog_features = utils.get_hog(resized_img, locations=locations,
                                    winSize=(self.roi_w_len, self.roi_h_len))
        hog_feature_list = list(utils.chunks(hog_features, int(len(hog_features) / len(locations))))

        predict_results = predict.clf.predict(hog_feature_list)

        predict.captured = False

        for i in range(0,len(predict_results)):
            if predict_results[i] == 0:
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
    app.run(host='localhost', port=8888)