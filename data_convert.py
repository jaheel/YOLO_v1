# -*- encoding: utf-8 -*-
'''
@File           : data_convert.py
@Time           : 2021/04/21 11:41:42
@Author         : Yang Hang (759979738@qq.com)
@Code Optimizer : Alex (xufanxin86@gmail.com)
@Version        : 1.0
'''

import xml.etree.ElementTree as ET
import os

DATASET_PATH = r"Dataset\VOC2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\\"

CLASSES = [
    "person",
    "bird",
    "cat",
    "cow",
    "dog",
    "horse",
    "sheep",
    "aeroplane",
    "bicycle",
    "boat",
    "bus",
    "car",
    "motorbike",
    "train",
    "bottle",
    "chair",
    "diningtable",
    "pottedplant",
    "sofa",
    "tvmonitor",
]

class Data2Label():
    def __init__(self, dataset_path, classes):
        """Data2Label init
        
        Parameters
        ----------
        dataset_path : {str, scalar}, the path to load dataset

        classes : {str, vector}, classes of the dataset
        """
        self.dataset_path = dataset_path
        self.classes = classes

    def __convert_bbox_format(self, img_size, box_points):
        """

        box_points(left upper, right bottom) format to (center_x, center_y, box_w, box_h),
        then normalized
        
        Parameters
        ----------
        img_size : {int, vector}, (img_width, img_height)

        box_points : {int, vector}, (xmin, ymin, xmax, ymax)

        Returns
        -------
        result : {float, vector}, (normalized_center_x, normalized_center_y, normalized_box_w, normalized_box_h)
        """
        
        dw = 1.0 / img_size[0]
        dh = 1.0 / img_size[1]

        center_x = (box_points[0] + box_points[2]) / 2.0
        center_y = (box_points[1] + box_points[3]) / 2.0
        
        box_w = box_points[2] - box_points[0]
        box_h = box_points[3] - box_points[1]
        
        center_x = center_x * dw
        center_y = center_y * dh
        box_w = box_w * dw
        box_h = box_h * dh

        return (center_x, center_y, box_w, box_h)

    def __convert_annotation(self, image_id):
        """
        
        convert image_id.xml to image_id.txt 
        which contain 
            the object class, 
            coordinate of the upper left corner in bbox (normalized)
            width and hight of the bbox (normalized)
        
        """
        
        in_file = open(self.dataset_path + "Annotations/%s" % (image_id))
        image_id = image_id.split(".")[0]

        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find("size")

        img_width = int(size.find("width").text)
        img_height = int(size.find("height").text)

        flag = 0

        for obj in root.iter("object"):
            difficult = obj.find("difficult").text
            cls_name = obj.find("name").text
            
            if cls_name not in self.classes or int(difficult) == 1:
                continue

            if flag == 0:
                out_file = open("./labels/%s.txt" % (image_id), "w+")
                flag = 1

            class_id = self.classes.index(cls_name)
            
            xmlbox = obj.find("bndbox")
            
            points = (
                float(xmlbox.find("xmin").text),
                float(xmlbox.find("ymin").text),
                float(xmlbox.find("xmax").text),
                float(xmlbox.find("ymax").text),
            )

            bbox = self.__convert_bbox_format((img_width, img_height), points)
            
            out_file.write(str(class_id) + " " + " ".join([str(temp) for temp in bbox]) + "\n")

    def make_label_txt(self):
        """
        
        create a file of image_id.txt in the labels folder, 
        one file of image_id.txt mapping one image_id.xml of bbox info 

        """
        filenames = os.listdir(self.dataset_path + "Annotations")
        for file in filenames:
            self.__convert_annotation(file)
    

data2label_model = Data2Label(dataset_path=DATASET_PATH, classes=CLASSES)
data2label_model.make_label_txt()
