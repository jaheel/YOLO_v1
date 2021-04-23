# -*- encoding: utf-8 -*-
'''
@File           : inference.py
@Time           : 2021/04/23 13:01:45
@Author         : Yang Hang (759979738@qq.com)
@Code Optimizer : Alex (xufanxin86@gmail.com)
@Version        : 1.0
'''

import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import torchvision.models as tvmodel
import os

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


class Voc2012(Dataset):
    def __init__(self, is_train=True):
        """

        Parameters
        ----------
        is_train : {bool (default : True), scalar}, 
                    if is train, load train.txt
                    else, load val.txt
        """

        self.file_name = []
        self.__file_name_init(is_train=is_train)

        self.img_path = r"./Dataset/VOC2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/"
        
        self.grid_S = 7
        self.grid_B = 2
        self.class_num = len(CLASSES)
        self.dim_len = 5 * self.grid_B + self.class_num
    

    def __file_name_init(self, is_train):
        """init file_name list

        Parameters
        ----------
        is_train : {bool, scalar}
        """
        if is_train:
            with open(r"./Dataset/VOC2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Main/train.txt", "r") as f:
                for x in f:
                    self.file_name.append(x.strip())
        
        else:
            with open(r"./Dataset/VOC2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Main/val.txt", "r") as f:
                for x in f:
                    self.file_name.append(x.strip())
    
    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, index):
        self.index = index
        
        img = cv2.imread(self.img_path + self.file_name[index] + ".jpg")

        img_height, img_width, _ = img.shape
        
        input_size = 448
        padw, padh = 0, 0

        if img_height > img_width:
            padw = (img_height - img_width) // 2
            img = np.pad(img, ((0, 0), (padw, padw), (0, 0)), "constant", constant_values=0)

        elif img_width > img_height:
            padh = (img_width - img_height) // 2
            img = np.pad(img, ((padh, padh), (0, 0), (0, 0)), "constant", constant_values=0)
        
        img = cv2.resize(img, (input_size, input_size))

        img_transform = transforms.Compose([transforms.ToTensor()])
        img = img_transform(img)
        
        with open("./labels/" + self.file_name[index] + ".txt", "r") as flabel:
            bbox = flabel.read().strip().split("\n")
            label = self.label_to_grid_label(bbox, padw, padh, img_width, img_height)
        
        label = torch.Tensor(label).cuda()

        return img, label
    

    def label_to_grid_label(self, bbox, padw, padh, img_width, img_height):
        """convert to the label in specific grid

        Parameters
        ----------
        bbox : {float, vector}, (class, cx, cy, box_w, box_h)

        padw : {int, scalar}

        padh : {int, scalar}

        img_width : {int, scalar}

        img_height : {int, scalar}

        Returns
        -------
        grid_label : {float, tensor} of shape (S, S, 5*B + C)
        """
        grid_label = np.zeros((self.grid_S, self.grid_S, self.dim_len))
        grid_size = 1.0 / self.grid_S

        for label in bbox:
            label = label.split(" ")
            label = [float(x.strip()) for x in label] #[class, cx, cy, w, h]

            if padw != 0:
                label[1] = (label[1] * img_width + padw) / img_height
                label[2] = (label[2] * img_width) / img_height
            
            elif padh != 0:
                label[3] = (label[3] * img_height + padh) / img_width
                label[4] = (label[4] * img_height) / img_width
            
            grid_x = int(label[1] // grid_size)
            grid_y = int(label[2] // grid_size)

            # center coordinate in the grid [1, gx, gy, w, h, 1, gx, gy, w, h, class[]]
            gx = (label[1] - grid_x * grid_size) / grid_size
            gy = (label[2] - grid_y * grid_size) / grid_size

            grid_label[grid_x, grid_y, 0 : 5 * self.grid_B] = np.array([1, gx, gy, label[3], label[4]] * self.grid_B)
            grid_label[grid_x, grid_y, 5 * self.grid_B + int(label[0])] = 1
        
        return grid_label


class Yolo_v1_resnet(nn.Module):
    def __init__(self):
        super(Yolo_v1_resnet, self).__init__()
        self.grid_S = 7
        self.grid_B = 2
        self.class_num = len(CLASSES)

        resnet = tvmodel.resnet34(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        
        self.last4_conv2 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, 2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, 1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, 1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 7 * 7 * 30),
            nn.Sigmoid(),
        )

    def forward(self, input):
        inputs = self.resnet(input)
        inputs = self.last4_conv2(inputs)
        inputs = inputs.view(inputs.size()[0], -1)
        inputs = self.fc(inputs)

        return inputs.reshape(-1, 7, 7, (5 * self.grid_B + self.class_num))




class Pred_to_BBox():
    def predtobbox(self, pred):
        ds = 1.0 / 7
    
        for i in range(7):
            for j in range(7):
                pred[i, j, 1] = j * ds + pred[i, j, 1] * ds
                pred[i, j, 6] = j * ds + pred[i, j, 6] * ds

                pred[i, j, 2] = i * ds + pred[i, j, 2] * ds
                pred[i, j, 7] = i * ds + pred[i, j, 7] * ds
    
        pred = pred.view(7 * 7, 30)

        return self.NMS(
            torch.cat(
                [
                    torch.cat([pred[:, 0:5], pred[:, 10:30]], dim=1),
                    torch.cat([pred[:, 5:10], pred[:, 10:30]], dim=1),
                ],
                dim=0,
            )
        )
    
    def NMS(self, bbox, conf_thresh=0.008, iou_thresh=0.1):
        bbox_cls_spec_conf = torch.mul(bbox[:, 5:], bbox[:, 0].unsqueeze(dim=1).expand(bbox[:, 5:].shape))

        bbox_cls_spec_conf[bbox_cls_spec_conf <= conf_thresh] = 0  # 将低于阈值的bbox忽略
        for cls20 in range(20):
            rank = torch.sort(bbox_cls_spec_conf[:, cls20], descending=True).indices
            for i in range(98):
                if bbox_cls_spec_conf[rank[i], cls20] != 0:
                    for j in range(i + 1, 98):
                        if bbox_cls_spec_conf[rank[j], cls20] != 0:
                            iou = self.calculate_iou(bbox[rank[i], 1:5], bbox[rank[j], 1:5])
                            if iou > iou_thresh:  # 根据iou进行非极大值抑制抑制
                                bbox_cls_spec_conf[rank[j], cls20] = 0
                    # print(bbox_cls_spec_conf[:, cls20])
        bbox = bbox[torch.max(bbox_cls_spec_conf, dim=1).values > 0]
        bbox_cls_spec_conf = bbox_cls_spec_conf[torch.max(bbox_cls_spec_conf, dim=1).values > 0]
        res = torch.ones((bbox.size()[0], 6))
        print(bbox)
        if bbox.size()[0] == 0:
            return res
        res[:, 1:5] = bbox[:, 1:5]
        res[:, 0] = torch.argmax(bbox[:, 5:], dim=1).int()
        res[:, 5] = torch.max(bbox_cls_spec_conf, dim=1).values  # 储存bbox对应的class-specific confidence scores
        print(res)
        return res

    def calculate_iou(self, cbbox1, cbbox2):
        pbbox1 = torch.zeros_like(cbbox1)
        pbbox2 = torch.zeros_like(cbbox2)
        pbbox1[0] = cbbox1[0] - 0.5 * cbbox1[2]
        pbbox1[2] = cbbox1[0] + 0.5 * cbbox1[2]
        pbbox1[1] = cbbox1[1] - 0.5 * cbbox1[3]
        pbbox1[3] = cbbox1[1] + 0.5 * cbbox1[3]

        pbbox2[0] = cbbox2[0] - 0.5 * cbbox2[2]
        pbbox2[2] = cbbox2[0] + 0.5 * cbbox2[2]
        pbbox2[1] = cbbox2[1] - 0.5 * cbbox2[3]
        pbbox2[3] = cbbox2[1] + 0.5 * cbbox2[3]
        intersect_bbox = [0.0, 0.0, 0.0, 0.0]

        if pbbox1[2] < pbbox2[0] or pbbox1[0] > pbbox2[2] or pbbox1[3] < pbbox2[1] or pbbox1[1] > pbbox2[3]:
            return 0
        else:
            intersect_bbox[0] = max(pbbox1[0], pbbox2[0])
            intersect_bbox[1] = max(pbbox1[1], pbbox2[1])
            intersect_bbox[2] = min(pbbox1[2], pbbox2[2])
            intersect_bbox[3] = min(pbbox1[3], pbbox2[3])

        area1 = (pbbox1[2] - pbbox1[0]) * (pbbox1[3] - pbbox1[1])  # bbox1面积
        area2 = (pbbox2[2] - pbbox2[0]) * (pbbox2[3] - pbbox2[1])  # bbox2面积
        area_intersect = (intersect_bbox[2] - intersect_bbox[0]) * (intersect_bbox[3] - intersect_bbox[1])  # 交集面积

        if area_intersect > 0:
            return area_intersect / (area1 + area2 - area_intersect)  # 计算iou
        else:
            return 0



COLOR = [
    (255, 0, 0),
    (255, 125, 0),
    (255, 255, 0),
    (255, 0, 125),
    (255, 0, 250),
    (255, 125, 125),
    (255, 125, 250),
    (125, 125, 0),
    (0, 255, 125),
    (255, 0, 0),
    (0, 0, 255),
    (125, 0, 255),
    (0, 125, 255),
    (0, 255, 255),
    (125, 125, 255),
    (0, 255, 0),
    (125, 255, 125),
    (255, 255, 255),
    (100, 100, 100),
    (0, 0, 0),
]


def draw_bbox(img, bbox):
    h, w = img.shape[0:2]
    n = bbox.size()[0]
    # print(bbox)
    img_ = img.copy()
    for i in range(n):
        p1 = (int(bbox[i, 1] * w - 0.5 * bbox[i, 3] * w), int(bbox[i, 2] * h - 0.5 * bbox[i, 4] * h))
        p2 = (int(bbox[i, 1] * w + 0.5 * bbox[i, 3] * w), int(bbox[i, 2] * h + 0.5 * bbox[i, 4] * h))
        cls_name = CLASSES[int(bbox[i, 0])]
        cv2.rectangle(img_, p1, p2, COLOR[int(bbox[i, 0])])
        cv2.putText(img_, cls_name, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    cv2.imshow("bbox", img_)
    cv2.waitKey(0)


if __name__ == "__main__":
    model = torch.load("./models_pkl/resnetYOLOv1_epoch100.pkl")
    val_dataloader = DataLoader(Voc2012(is_train=True), batch_size=1, shuffle=False)
    for i, (inputs, labels) in enumerate(val_dataloader):
        inputs = inputs.cuda()
        
        pred = model(inputs)
        pred = pred.squeeze(dim=0)
        # FF = pred.cpu().detach().numpy()
        # np.save("ff.npy", FF)
        labels = labels.squeeze(dim=0)
        pred_to_bbox_model = Pred_to_BBox()
        bbox = pred_to_bbox_model.predtobbox(pred)
        inputs = inputs.squeeze(dim=0)  # 输入图像的尺寸是(1,3,448,448),压缩为(3,448,448)
        inputs = inputs.permute((1, 2, 0))  # 转换为(448,448,3)
        img = inputs.cpu().numpy()
        img = 255 * img  # 将图像的数值从(0,1)映射到(0,255)并转为非负整形
        img = img.astype(np.uint8)
        draw_bbox(img, bbox.cpu())  # 将网络预测结果进行可视化，将bbox画在原图中，可以很直观的观察结果
        print(bbox.size(), bbox)