__author__ = 'xiehaoina'
import time
import requests
import base64
import os

if __name__ == '__main__':
    dir = "../../hog-based-object-detection/test"
    for img in os.listdir(dir):
        if not img.endswith("jpeg"):
            continue
        else:
            file = os.path.join(dir,img)
        print(file)
        with open(file,"rb" ) as f:
            base64_img = base64.b64encode(f.read())
            url = 'http://localhost:10001'
            params = {
                "client_id": "123",
                "params": [
                    {
                        "id": "1",
                        "type": "jpeg",
                        "data": base64_img.decode("utf8"),
                    }
                ]
            }
            start_time = time.time()
            #for i in range(0,1000):
            r = requests.post(url, json = params)
            print(r.json())
            end_time = time.time()
            #print("start:{},end:{}".format(start_time, end_time))
            #print(1000/(end_time-start_time))