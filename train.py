# -*- encoding: utf-8 -*-
'''
@File           : train.py
@Time           : 2021/04/22 18:31:45
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

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

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


class Yolo_v1_loss(nn.Module):
    def __init__(self):
        super(Yolo_v1_loss, self).__init__()
        self.grid_S = 7
        self.grid_B = 2
        self.class_num = len(CLASSES)
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.dim_len = 5 * self.grid_B + self.class_num

    def calculate_iou(self, cprebbox, clabelbbox):
        """

        Parameters
        ----------
        cprebbox : {float, tensor(3-dim)} of shape (true_label_num, grid_B, [bbox_cx, bbox_cy, bbox_w, bbox_h])
        
        clabelbbox : {float, tensor(3-dim)}of shape (true_label_num, grid_B, [bbox_cx, bbox_cy, bbox_w, bbox_h])
        
        Returns
        -------
        iou_mat : {bool, matrix}, (true_label_num, grid_B), 
                    the selected bbox in grid will be True, other is False
        
        iou : {float, matrix}, (true_label_num, grid_B)
        """
        
        # prebbox : {float, tensor(3-dim)}, (true_label_num, grid_B, [bbox_x1, bbox_y1, bbox_x2, bbox_y2])
        prebbox = torch.zeros_like(cprebbox)
        
        prebbox[:, :, 0] = cprebbox[:, :, 0] - 0.5 * cprebbox[:, :, 2]
        prebbox[:, :, 1] = cprebbox[:, :, 1] - 0.5 * cprebbox[:, :, 3]
        prebbox[:, :, 2] = cprebbox[:, :, 0] + 0.5 * cprebbox[:, :, 2]
        prebbox[:, :, 3] = cprebbox[:, :, 1] + 0.5 * cprebbox[:, :, 3]

        # labelbbox : {float, tensor(3-dim)}, (true_label_num, grid_B, [bbox_x1, bbox_y1, bbox_x2, bbox_y2])
        labelbbox = torch.zeros_like(clabelbbox)

        labelbbox[:, :, 0] = clabelbbox[:, :, 0] - 0.5 * clabelbbox[:, :, 2]
        labelbbox[:, :, 1] = clabelbbox[:, :, 1] - 0.5 * clabelbbox[:, :, 3]
        labelbbox[:, :, 2] = clabelbbox[:, :, 0] + 0.5 * clabelbbox[:, :, 2]
        labelbbox[:, :, 3] = clabelbbox[:, :, 1] + 0.5 * clabelbbox[:, :, 3]

        # find the intersection whether exist between the two bbox
        mat1 = prebbox[:, :, 2] < labelbbox[:, :, 0] # pred_bbox_x2 < label_bbox_x1
        mat2 = prebbox[:, :, 0] > labelbbox[:, :, 2] # pred_bbox_x1 > label_bbox_x2
        mat3 = prebbox[:, :, 3] < labelbbox[:, :, 1] # pred_bbox_y2 < label_bbox_y1
        mat4 = prebbox[:, :, 1] > labelbbox[:, :, 3] # pred_bbox_y1 > label_bbox_y2
        mat = (mat1 | mat2 | mat3 | mat4) == False # exist intersection

        # intersection area
        # intersect_bbox : {float, tensor(3-dim)}, (true_label_num, grid_B, [bbox_x1, bbox_y1, bbox_x2, bbox_y2])
        intersect_bbox = torch.zeros_like(cprebbox)

        intersect_bbox[:, :, 0] = torch.max(torch.stack([prebbox[:, :, 0], labelbbox[:, :, 0]], dim=2), dim=2).values
        intersect_bbox[:, :, 1] = torch.max(torch.stack([prebbox[:, :, 1], labelbbox[:, :, 1]], dim=2), dim=2).values
        intersect_bbox[:, :, 2] = torch.min(torch.stack([prebbox[:, :, 2], labelbbox[:, :, 2]], dim=2), dim=2).values
        intersect_bbox[:, :, 3] = torch.min(torch.stack([prebbox[:, :, 3], labelbbox[:, :, 3]], dim=2), dim=2).values

        # prebbox_area : {float, matrix}, (true_label_num, grid_B)
        prebbox_area = torch.mul(prebbox[:, :, 2] - prebbox[:, :, 0], prebbox[:, :, 3] - prebbox[:, :, 1])

        # labelbbox_area : {float, matrix}, (true_label_num, grid_B)
        labelbbox_area = torch.mul(labelbbox[:, :, 2] - labelbbox[:, :, 0], labelbbox[:, :, 3] - labelbbox[:, :, 1])

        # intersect_area : {float, matrix}, (true_label_num, grid_B)
        intersect_area = torch.mul(intersect_bbox[:, :, 2] - intersect_bbox[:, :, 0], intersect_bbox[:, :, 3] - intersect_bbox[:, :, 1])
        intersect_area = torch.mul(intersect_area, mat)

        # iou : {float, matrix}, (true_label_num, grid_B)
        iou = intersect_area / (prebbox_area + labelbbox_area - intersect_area)
        
        # iou_indices : {int, vector}, (true_label_num), the max iou of bbox indices in single grid
        iou_indices = iou.max(dim=1).indices 

        # iou_mat : {bool, matrix}, (true_label_num, grid_B)
        iou_mat = torch.zeros_like(iou) == 1

        iou_indices = torch.stack([(torch.arange(len(iou_indices))).cuda(), iou_indices], dim=1)
        iou_mat[iou_indices[:, 0], iou_indices[:, 1]] = True

        return iou_mat, iou

    def forward(self, predict, labels):
        """

        Parameters
        ----------
        predict : {float, tensor(4-dim)} of shape (batch_size, grid_S, grid_S, 5 * B + C)

        labels : {float, tensor(4-dim)} of shape (batch_size, grid_S, grid_S, 5 * B + C)
        
        Returns
        -------
        loss : {float, scalar}, total loss / batch_size

        obj_class_loss : {float, scalar}, object class loss / batch_size

        obj_confidence_loss : {float, scalar}, object confidence total loss / batch_size

        obj_coord_x_y_w_h_loss : {float, scalar}, lambda coord with xywh total loss / batch_size

        noobj_loss : {float, scalar}, no object loss / batch_size

        """
        
        # cover_mat : {int, tensor(3-dim)} of shape (batch_size, grid_S, grid_S)
        cover_mat = torch.arange(self.grid_S).cuda()
        cover_mat = cover_mat.expand(predict.shape[0], self.grid_S, self.grid_S)

        predict_copy = predict.clone()
        labels_copy = labels.clone()

        # relative coordinates convert to absolute coordinates(normalized)
        # ------------------------------------
        predict_copy[:, :, :, 1] = (predict_copy[:, :, :, 1] + cover_mat) / self.grid_S
        predict_copy[:, :, :, 6] = (predict_copy[:, :, :, 6] + cover_mat) / self.grid_S
        labels_copy[:, :, :, 1] = (labels_copy[:, :, :, 1] + cover_mat) / self.grid_S
        labels_copy[:, :, :, 6] = (labels_copy[:, :, :, 6] + cover_mat) / self.grid_S

        # cover_matx : {int, tensor(3-dim)} of shape (batch_size, grid_S, grid_S)
        cover_matx = cover_mat[0, :, :].t().unsqueeze(0).expand(predict.shape[0], predict.shape[1], predict.shape[2])

        predict_copy[:, :, :, 2] = (predict_copy[:, :, :, 2] + cover_matx) / self.grid_S
        predict_copy[:, :, :, 7] = (predict_copy[:, :, :, 7] + cover_matx) / self.grid_S
        labels_copy[:, :, :, 2] = (labels_copy[:, :, :, 2] + cover_matx) / self.grid_S
        labels_copy[:, :, :, 7] = (labels_copy[:, :, :, 7] + cover_matx) / self.grid_S
        # ------------------------------------

        # label_class : {int, tensor(4-dim)} of shape (batch_size, grid_S, grid_S, 1)
        label_class = labels[:, :, :, 0].unsqueeze(-1)
        label_class = label_class.expand(predict.shape) == 1

        # absolute coordinate
        # -------------------------------------
        # has object (predict)
        obj_i_pred = torch.masked_select(predict_copy, label_class).reshape(-1, self.dim_len)
        # obj_bbox_pred : {float, tensor(3-dim)} of shape (true_label_num, grid_B, 5)
        obj_bbox_pred = obj_i_pred[:, 0 : 5 * self.grid_B].reshape(-1, self.grid_B, 5)

        # has object (label)
        obj_i_label = torch.masked_select(labels_copy, label_class).reshape(-1, self.dim_len)
        obj_bbox_label = obj_i_label[:, 0 : 5 * self.grid_B].reshape(-1, self.grid_B, 5)
        
        iou_mat, iou = self.calculate_iou(obj_bbox_pred[:, :, 1:5], obj_bbox_label[:, :, 1:5])
        # -------------------------------------

        # relative coordinate
        # -------------------------------------
        no_copy_obj_i_pred = torch.masked_select(predict, label_class).reshape(-1, self.dim_len)
        no_copy_obj_bbox_pred = no_copy_obj_i_pred[:, 0 : 5 * self.grid_B].reshape(-1, self.grid_B, 5)

        no_copy_obj_i_label = torch.masked_select(labels, label_class).reshape(-1, self.dim_len)
        no_copy_obj_bbox_label = no_copy_obj_i_label[:, 0 : 5 * self.grid_B].reshape(-1, self.grid_B, 5)
        # -------------------------------------

        # true_bbox_label : {float, tensor(3-dim)} of shape (true_label_num, grid_B, [confidence, cx, cy, w, h])
        true_bbox_label = torch.cat([iou.unsqueeze(2), no_copy_obj_bbox_label[:, :, 1:]], dim=2)

        # iou_mat : {int, tensor(3-dim)} of shape (true_label_num, grid_B) -> (true_label_num, grid_B, 5)
        iou_mat = iou_mat.unsqueeze(2).expand(obj_bbox_pred.shape)
        
        # select the bbox which has the max iou in every grid
        # obj_ij_xxx : 
        #           (true_label_num, grid_B, [confidence, cx, cy, w, h]) 
        #               -> (true_label_num, 1, [confidence, cx, cy, w, h])
        #               -> (true_label_num, [confidence, cx, cy, w, h])
        obj_ij_pred = torch.masked_select(no_copy_obj_bbox_pred, iou_mat).reshape(-1, 5)
        obj_ij_label = torch.masked_select(true_bbox_label, iou_mat).reshape(-1, 5)
        
        # sqrt(w), sqrt(h)
        obj_ij_pred[:, 3:5] = torch.sqrt(obj_ij_pred[:, 3:5])
        obj_ij_label[:, 3:5] = torch.sqrt(obj_ij_label[:, 3:5])

        # obj_ij_xxx:
        #           (true_label_num, [confidence, cx, cy, w, h])
        #               -> (true_label_num, [confidence, cx, cy, w, h, class[] ])
        obj_ij_pred = torch.cat([obj_ij_pred, obj_i_pred[:, 5 * self.grid_B : self.dim_len]], dim=1)
        obj_ij_label = torch.cat([obj_ij_label, obj_i_label[:, 5 * self.grid_B : self.dim_len]], dim=1)

        # ----------------------------------
        # calculate the loss
        # 
        # part 1 : has object loss
        #
        # obj_c_x_y_w_h_c_loss : {float, vector} of shape [confidence_loss, x_loss, y_loss, w_loss, h_loss, class1_loss, ... , classC_loss]
        obj_c_x_y_w_h_c_loss = torch.square(obj_ij_label - obj_ij_pred).sum(dim=0)

        obj_class_loss = obj_c_x_y_w_h_c_loss[5:].sum()
        obj_confidence_loss = obj_c_x_y_w_h_c_loss[0]
        obj_coord_x_y_w_h_loss = self.lambda_coord * (obj_c_x_y_w_h_c_loss[1:5].sum())

        # part 1.1 : object total loss
        obj_loss = obj_coord_x_y_w_h_loss + obj_class_loss + obj_confidence_loss
        
        #
        # part 2 : no object loss
        #
        
        noobj_iou_mat = iou_mat == False

        # part 2.1 : label is 1
        # noobj just compute the confidence loss, so obj_bbox_pred and no_copy_obj_bbox_pred both can do 
        noobj_ij_pred = torch.masked_select(obj_bbox_pred, noobj_iou_mat).reshape(-1, 5)
        noobj_ij_label = torch.masked_select(true_bbox_label, noobj_iou_mat).reshape(-1, 5)
        
        # compute the confidence loss
        noobj_ij_pred = noobj_ij_pred[:, 0]
        noobj_ij_label = noobj_ij_label[:, 0]
        
        noobj_label1_loss = self.lambda_noobj * torch.square(noobj_ij_label - noobj_ij_pred).sum()

        # part 2.2 : label is 0
        no_label_class = label_class == False
        
        noobj_i_pred = torch.masked_select(predict_copy, no_label_class).reshape(-1, self.dim_len)

        noobj_i_pred_confidence = torch.stack([noobj_i_pred[:, 0], noobj_i_pred[:, 5]], dim=1)
        noobj_i_pred_confidence = torch.square(noobj_i_pred_confidence)

        noobj_label0_loss = self.lambda_noobj * noobj_i_pred_confidence.sum()

        # part 2.3 : no object total loss
        noobj_loss = noobj_label1_loss + noobj_label0_loss

        # part 3 : total loss
        loss = obj_loss + noobj_loss
        # ----------------------------------

        return (
            loss / predict.shape[0],
            obj_class_loss / predict.shape[0],
            obj_confidence_loss / predict.shape[0],
            obj_coord_x_y_w_h_loss / predict.shape[0],
            noobj_loss / predict.shape[0],
            )



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

if __name__ == "__main__":
    epoch = 100
    batch_size = 10
    lr = 0.00001

    train_data = Voc2012()
    train_dataloader = DataLoader(Voc2012(is_train=True), batch_size=batch_size, shuffle=True)
    
    model = Yolo_v1_resnet().cuda()

    # more GPU parallel
    model = nn.DataParallel(model)
    model = model.cuda()

    # for layer in model.children():
    #     layer.requires_grad = False
    #     break

    criterion = Yolo_v1_loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005)

    total_loss_list = []
    epoch_loss_avg_list = []

    for e in range(epoch):
        model.train()
        loss_list = []

        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.cuda()
            labels = labels.float().cuda()
            predict = model(inputs)

            loss, obj_class_loss, obj_confidence_loss, obj_coord_x_y_w_h_loss, noobj_loss = criterion(predict, labels)

            loss_list.append(loss)
            total_loss_list.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                "Epoch %d/%d | Step %d/%d | Loss: %.2f | class_loss: %.2f | confidence_loss: %.2f | coord_xywh_loss: %.2f | noobj_loss: %.2f "
                % (e+1, epoch, i, len(train_data) // batch_size, loss, obj_class_loss, obj_confidence_loss, obj_coord_x_y_w_h_loss, noobj_loss)
            )
        
        epoch_loss_avg_list.append(np.sum(loss_list) / len(loss_list))

        np.save("./loss/epoch" + str(e) + "batch_loss.npy", epoch_loss_avg_list)
        np.save("batch_loss.npy", total_loss_list)

        if (e + 1) % 20 == 0:
            torch.save(model, "./models_pkl/resnetYOLOv1_epoch" + str(e+1) + ".pkl")
