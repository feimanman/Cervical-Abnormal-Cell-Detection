from models.basic_block import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
import torch
import cfg
from torch.autograd import Variable
def clipBoxes1(boxes):

    height, width = cfg.patch_size

    boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
    boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

    boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
    boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

    return boxes

import cv2
from losses1 import calc_iou1
import numpy as np
from torchvision.ops import nms
def croppatch(classification, regression,anchors,inputs,annotations,idx,imgpath,org_path,epoch):

    bbox_id_all = []
    label=[]
    imgName = []
    image=[]
    score_all=[]
    result_all = []
    in_size = inputs.size()[2]
    unit = torch.stack([torch.arange(0, in_size)] * in_size).float()
    x = torch.stack([unit.t()] * 3)
    y = torch.stack([unit] * 3)
    if isinstance(inputs, torch.cuda.FloatTensor):
        x, y = x.cuda(), y.cuda()
    IOU_area = []
    for i0 in range(inputs.size(0)):
        classification1 =Variable(classification[i0].unsqueeze(0))
        regression1 = Variable(regression[i0].unsqueeze(0))
        regressBoxes = BBoxTransform()
        transformed_all = []
        bbox_id = []
        transformed_anchors = regressBoxes(anchors, regression1)
        transformed_anchors = clipBoxes1(transformed_anchors)
        for class_i in range(classification1.shape[-1]):
            scores = torch.squeeze(classification1[:,class_i])
            scores_over_thresh = (scores > 0.5)
            transformed_anchors_1 = transformed_anchors[0, scores_over_thresh, :]
            scores = scores[0, scores_over_thresh, :]
            transformed_all.append(torch.cat([transformed_anchors_1, scores], dim=1))
            if bool(transformed_all):
                transformed_all = torch.cat(transformed_all, dim=0)
                if transformed_all.size(0) == 0:
                    print('error')
                else:
                    anchors_num_idx = nms(transformed_all, 0.05)
                    if len(anchors_num_idx) < cfg.patch_num:
                        print('error', len(anchors_num_idx))
                        patch_num = len(anchors_num_idx)
                    else:
                        patch_num = cfg.patch_num

                    for i in range(patch_num):
                        tx = int(transformed_anchors_1[anchors_num_idx[i], 0])
                        ty = int(transformed_anchors_1[anchors_num_idx[i], 1])
                        tx1 = int(transformed_anchors_1[anchors_num_idx[i], 2])
                        ty1 = int(transformed_anchors_1[anchors_num_idx[i], 3])
                        p=float(scores[anchors_num_idx[i], 0])
                        bbox_id.append([class_i,d[0][anchors_num_idx[i]]])
                        size = 256
                        criterion = int(size / 2)
                        x0 = int((tx + tx1) / 2)
                        y0 = int((ty + ty1) / 2)
                        up = y0
                        low = 1024 - y0
                        left = x0
                        right = 1024 - x0
                        if up < criterion:
                            up_d = 0
                            low_d = size
                        elif low < criterion:
                            low_d = 1024
                            up_d = 1024 - size
                        else:
                            up_d = y0 - criterion
                            low_d = y0 + criterion
                        if left < criterion:
                            left_d = 0
                            right_d = size
                        elif right < criterion:
                            right_d = 1024
                            left_d = 1024 - size
                        else:
                            left_d = x0 - criterion
                            right_d = x0 + criterion
                        GT_image = inputs[i0].detach().cpu().numpy().transpose(1, 2, 0)
                        GT_image = GT_image * 255
                        wrimg=GT_image[up_d:low_d, left_d:right_d, :]

                        left_d2 = torch.tensor(tx, dtype=torch.float).cuda()
                        up_d2 = torch.tensor(ty, dtype=torch.float).cuda()
                        right_d2 = torch.tensor(tx1, dtype=torch.float).cuda()
                        low_d2 = torch.tensor(ty1, dtype=torch.float).cuda()
                        pos2= [left_d2, up_d2, right_d2, low_d2]

                        annotations1 = annotations[i0, :, :]
                        IOU= calc_iou1(pos2, annotations1)

                        IOU_area.append(IOU)

                        if IOU > 0.7:
                            label.append([class_i])
                            xmm=class_i
                        else:
                            label.append([5])
                            xmm=5

                        cv2.imwrite(imgpath+org_path[i0]+'_%02d_%06d_%02d_%02d_%02d.jpg'%(epoch,idx,i,class_i,xmm),wrimg)
                        imgname=org_path[i0]+'_%02d_%06d_%02d_%02d_%02d.jpg'%(epoch,idx,i,class_i,xmm)
                        imgName.append(imgname)
                        image.append(wrimg)
                        score_all.append(p)
                        result_all.append(pos2)

            bbox_id_all.append(bbox_id)


    return label,bbox_id_all,imgName,result_all

