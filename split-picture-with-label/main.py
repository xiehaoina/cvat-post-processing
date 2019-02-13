__author__ = 'xiehaoina'
from optparse import OptionParser
import xmltodict
import os
import cv2
import numpy as np

def split_images(img, picture_info,  width , height):
    split_images = {}
    img = np.pad(img, ((height / 2, height / 2), (width / 2, width / 2), (0,0)), 'edge')
    for points in picture_info:
        p_data = points['@points']
        if points['@label'] not in split_images:
            split_images[points['@label']] = []
        p_w_data, p_h_data = map(int, map(float,p_data.split(",")))
        data = img[p_h_data : p_h_data + height  , p_w_data :  p_w_data + width ]
        split_images[points['@label']].append(data)
    return split_images

def parse_cvat(config):
    dict = xmltodict.parse(config)
    if "annotations" in dict and "image" in dict["annotations"]:
        return dict["annotations"]["image"]
    else:
        return None

def save_images(images , output_dir, prefix_name):
    for k,v in images.items():
        dest_dir = os.path.join(output_dir,k)
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        for i, val in enumerate(v):
            dest_img = os.path.join(dest_dir, prefix_name.split(".")[0] + "_"  + str(i) + ".jpg")
            cv2.imwrite(dest_img, val)

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
        "-o", "--output",
        action="store",
        dest="output",
        type="string",
        default="output",
        help="specify dir that storing output pictures",
        metavar = "OUTPUT"
    )

    parser.add_option(
        "-c", "--config",
        action="store",
        dest="config",
        type="string",
        default="cvat.xml",
        help="specify configs that generated by cvat",
        metavar = "CONFIG"
    )

    parser.add_option(
        "-w", "--width",
        action="store",
        dest="width",
        type="int",
        default="400",
        help="specify the width of splited picture",
        metavar = "WIDTH"
    )

    parser.add_option(
        "-e", "--height",
        action="store",
        dest="height",
        type="int",
        default="300",
        help="specify the height of splited picture",
        metavar = "HEIGHT"
    )

    (options, args) = parser.parse_args()

    if not os.path.isfile(options.config):
        print("{0} is not file".format(options.config))
        exit(0)
    with open(options.config, 'r') as f:
        cvat_results = parse_cvat(f.read())

    if cvat_results is None:
        print("get a empty parsed result from {0}".format(options.config))
        exit(0)

    if not os.path.isdir(options.input) or not os.path.isdir(options.output):
        print("{0} or {1} is not a dir".format(options.input,options.output))
        exit(0)

    for img in cvat_results:
        img_path = os.path.join(options.input, img['@name'])
        img_data = cv2.imread(img_path)
        labeled_pictures = split_images(img_data, img['points'],  options.width, options.height)
        save_images(labeled_pictures, options.output, img['@name'])
