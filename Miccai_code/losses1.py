import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basic_block import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from torch.autograd import Variable, Function
import cfg

def py_cpu_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    # print('-------------scores',scores)
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]  # because index start from 1

    return keep


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua
    return IoU
def calc_iou0(a, b):
    area = (b[ 2] - b[ 0]) * (b[3] - b[ 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[ 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    # ua = torch.min(torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1),area)

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua
    return IoU

def calc_iou1(a, b):
    ixmin = torch.max(b[:, 0], a[0])
    iymin = torch.max(b[:, 1], a[1])
    ixmax = torch.min(b[:, 2], a[2])
    iymax = torch.min(b[:, 3], a[3])
    iw=(ixmax-ixmin+1)
    ih=(iymax-iymin+1)
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    # iw = torch.max(ixmax - ixmin + 1., torch.tensor(0,dtype=float))
    # # overlap area height
    # ih = torch.max(iymax - iymin + 1., torch.tensor(0,dtype=float))
    # overlap areas
    inters = iw * ih
    ## calculate ious
    # union
    # print('a[2] - a[0] ',a[2] - a[0])
    # print('a[3] - a[1]',a[3] - a[1])
    # print('b[:, 2] - b[:, 0]',b[:, 2] - b[:, 0])
    # print('b[:, 3] - b[:, 1]',b[:, 3] - b[:, 1])
    # union = ((abs(a[2] - a[0]) + 1.) * (abs(a[3] - a[1]) + 1.) + \
    #          (b[:, 2] - b[:, 0] + 1.) * (b[:, 3] - b[:, 1] + 1.) - \
    #          inters)
    union = torch.min((abs(a[2] - a[0]) + 1.) * (abs(a[3] - a[1]) + 1.),(b[:, 2] - b[:, 0] + 1.) * (b[:, 3] - b[:, 1] + 1.))
    # ious
    # print('%.3f/%.3f'%(inters,union))
    # if union<0:
    #     print('all_area',(abs(a[2] - a[0]) + 1.) * (abs(a[3] - a[1]) + 1.) + \
    #          (b[:, 2] - b[:, 0] + 1.) * (b[:, 3] - b[:, 1] + 1.) )
    #     print('iter',inters)
    #     print(ixmin,ixmax,iymin,iymax)
    #     print('a',a)
    #     print('b',b)
    #     print('a[2] - a[0] ',abs(a[2] - a[0]))
    #     print('a[3] - a[1]',abs(a[3] - a[1]))
    #     print('b[:, 2] - b[:, 0]',abs(b[:, 2] - b[:, 0]))
    #     print('b[:, 3] - b[:, 1]',abs(b[:, 3] - b[:, 1]))
    overlaps = inters / union
    # print(overlaps)
    # print(torch.max(overlaps))
    # find the maximum iou
    ovmax = torch.max(overlaps)
    # print('overlaps',overlaps)
    # print('ovmax',ovmax)
    # IoU_max_anchor, IoU_argmax_anchor = torch.max(overlaps, dim=0)

    # print('IoU_max_anchor',IoU_max_anchor)
    # print('IoU_argmax_anchor',IoU_argmax_anchor)
    # print('b[IoU_argmax_anchor]',b[IoU_argmax_anchor])
    # print('a',a)
    # if ovmax>0.2:
    #     print('all_area',(abs(a[2] - a[0]) + 1.) * (abs(a[3] - a[1]) + 1.) + \
    #          (b[:, 2] - b[:, 0] + 1.) * (b[:, 3] - b[:, 1] + 1.) )
    #     print('iter',inters)
    #     print(ixmin,ixmax,iymin,iymax)
    #     print('a',a)
    #     print('b',b)

    # find the maximum iou index
    # jmax = torch.argmax(overlaps)
    # area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    #
    # iw = torch.min(a[2], b[:, 2]) - torch.max(a[0], b[:, 0])
    # ih = torch.min(a[3], b[:, 3]) - torch.max(a[1], b[:, 1])
    #
    # iw = torch.clamp(iw, min=0)
    # ih = torch.clamp(ih, min=0)
    #
    # ua = (a[2] - a[ 0]) * (a[3] - a[1]) + area - iw * ih
    #
    # ua = torch.clamp(ua, min=1e-8)
    #
    # intersection = iw * ih
    #
    # IoU = intersection / ua
    # return ovmax,a,b[IoU_argmax_anchor]
    return ovmax

def clipBoxes1(boxes):

    height, width = cfg.patch_size

    boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
    boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

    boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
    boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)
    # print('--------------------------------clipbox')
    return boxes

class RankLoss(nn.Module):
    def __init__(self,alpha=0.25, gamma=2.0):
        super(RankLoss, self).__init__()

    def forward(self,patch_score,classifications,bbox_id,batch_num,regressions, anchors, annotations,label,patch_img):

        loss = []
        bceloss=[]
        loss_all=[]
        i=0
        regression_losses = []
        anchor = anchors[0, :, :]
        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights
        numx=0
        pos_zuobiao=[]
        for j in range(batch_num):
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            regression_pre = regressions[j, :, :].unsqueeze(0)
            regressBoxes = BBoxTransform()
            transformed_anchors = regressBoxes(anchors, regression_pre)
            classification = torch.clamp(classification, 1e-5, 1.0 - 1e-5)
            positive_indices = torch.zeros(classification.shape[0]).type(torch.bool)
            bce_all=[]
            for id in range(len(bbox_id[j])):

                if label[numx][0]==0:
                    supressloss =(patch_score[i][1]- (1 - classification[bbox_id[j][id]]) + 0.05).clamp(min=torch.tensor(0).cuda().float()) -torch.log(1.0 - classification[bbox_id[j][id]])*0.5
                    loss.append(supressloss)
                else:
                    supressloss=(patch_score[i][0]-classification[bbox_id[j][id]] + 0.05).clamp(min =torch.tensor(0).cuda().float())
                    loss.append(supressloss)
                if label[numx][0]==1:
                    positive_indices[bbox_id[j][id]] = True

                xx_id = []
                IoU_anchor = calc_iou0(anchors[0, :, :], patch_img[i])  # num_anchors x num_annotations
                IoU_max_anchor, IoU_argmax_anchor = torch.max(IoU_anchor, dim=1)
                target_anchor = torch.ones(classification.shape) * -1
                target_anchor = target_anchor.cuda()
                target_anchor[torch.lt(IoU_max_anchor, 0.4), 0] = 0.0
                target_anchor[torch.ge(IoU_max_anchor, 0.5), 0] = 1.0
                positive_indices_anchor = torch.ge(IoU_max_anchor, 0.6)
                for xx, e in enumerate(positive_indices_anchor.cpu().tolist()):
                    if e:
                        if xx != bbox_id[j][id]:
                            xx_id.append(xx)

                for cls_id in range(len(xx_id)):
                    if label[numx][0]==0:
                        supressloss =(patch_score[i][1]- (1 - classification[xx_id[cls_id]]) + 0.05).clamp(min=torch.tensor(0).cuda().float())-torch.log(1.0 - classification[xx_id[cls_id]])
                        loss.append(supressloss)
                    else:
                        supressloss=(patch_score[i][0]-classification[xx_id[cls_id]] + 0.05).clamp(min =torch.tensor(0).cuda().float())
                        loss.append(supressloss)

                i=i+1

                numx=numx+1


            bbox_annotation = annotations[j, :, :]  ###gt

            bbox_annotation = bbox_annotation[bbox_annotation[:, 0] != -1]


            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().cuda())
                bceloss.append(torch.tensor(0.0).float().cuda())
                continue

            if len(bbox_id[j])==0:
                loss_all.append(torch.tensor(0).cuda().float())
                bceloss.append(torch.tensor(0).cuda().float())
            else:
                loss_all.append(torch.sum(torch.stack(loss))/(len(xx_id)+len(bbox_id[j])))
                bceloss.append(torch.sum(torch.stack(bce_all))/len(bbox_id[j]))

            IoU = calc_iou(anchors[0, :, :], bbox_annotation)  # num_anchors x num_annotations
            IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1
            assigned_annotations = bbox_annotation[IoU_argmax, :]
            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]
                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]
                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)
                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )

                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(torch.tensor(0).cuda().float())

        rank_indices = len(loss)

        if rank_indices==0:
            rank_indices=1

        loss=torch.sum(torch.stack(loss))

        return loss/rank_indices,torch.stack(regression_losses).mean(dim=0, keepdim=True)

import torch.nn as nn
from utils import *

class ContrastLoss(nn.Module):
    def __init__(self):
        super(ContrastLoss, self).__init__()

    def forward(self,ROI_features,patch_features):
        n = ROI_features.shape[0]
        similarity = ROI_features.mm(ROI_features.t())
        norm = torch.reshape(torch.norm(similarity, 2, 1), (-1, 1))
        norm_similarity = similarity / norm

        ema_similarity = patch_features.mm(patch_features.t())
        ema_norm = torch.reshape(torch.norm(ema_similarity, 2, 1), (-1, 1))
        ema_norm_similarity = ema_similarity / ema_norm

        similarity_mse_loss = (norm_similarity - ema_norm_similarity) ** 2

        return torch.sum(similarity_mse_loss)/n


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = self.alpha
        gamma = self.gamma

        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]   ###gt
            bbox_annotation = bbox_annotation[bbox_annotation[:, 0] != -1]
            # if bbox_annotation.shape[0] == 0:
            #     regression_losses.append(torch.tensor(0).float().cuda())
            #     classification_losses.append(torch.tensor(0).float().cuda())
            #
            #     continue
            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(classification.shape).cuda() * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    print('np label  clsloss', cls_loss.sum())
                    regression_losses.append(torch.tensor(0).cuda().float())

                else:
                    alpha_factor = torch.ones(classification.shape) * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce

                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())

                continue

            classification = torch.clamp(classification, 1e-5, 1.0 - 1e-5)
            IoU = calc_iou(anchors[0, :, :], bbox_annotation)  # num_anchors x num_annotations
                           ###pre             gt
            IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1
            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1
            targets = targets.cuda()
            targets[torch.lt(IoU_max, 0.4), 0] = 0.0
            targets[torch.ge(IoU_max, 0.5), 0] = 1.0
            negative_indices = torch.lt(IoU_max, 0.4)
            num_negative_anchors = negative_indices.sum()

            positive_indices = torch.ge(IoU_max, 0.5)
            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]
            # targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
            alpha_factor = torch.ones(targets.shape) * alpha
            alpha_factor = alpha_factor.cuda()

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            # focal_weight = torch.clamp(focal_weight, 1e-4, 1.0)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce
            # print('clsloss',cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()

                negative_indices = ~positive_indices

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                # print('resloss',regression_loss.mean())
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(torch.tensor(0).cuda().float())
        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
               torch.stack(regression_losses).mean(dim=0, keepdim=True)
class FocalLoss_val(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss_val, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = self.alpha
        gamma = self.gamma
        regression_losses = []
        classification_losses = []
        regression_losses = []
        anchor = anchors[:, :]

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights


        classification = classifications[:, :]
        regression = regressions[ :, :]

        bbox_annotation = annotations[ :, :]
        bbox_annotation = bbox_annotation[bbox_annotation[:, 0] != -1]
        if bbox_annotation.shape[0] == 0:
            regression_losses.append(torch.tensor(0).float().cuda())
            classification_losses.append(torch.tensor(0).float().cuda())


        classification = torch.clamp(classification, 1e-5, 1.0 - 1e-5)
        IoU = calc_iou(anchors[0, :, :], bbox_annotation)  # num_anchors x num_annotations

        IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1
        # compute the loss for classification
        targets = torch.ones(classification.shape) * -1
        targets = targets.cuda()
        targets[torch.lt(IoU_max, 0.4), 0] = 0.0
        targets[torch.ge(IoU_max, 0.5), 0] = 1.0

        negative_indices = torch.lt(IoU_max, 0.4)
        num_negative_anchors = negative_indices.sum()

        positive_indices = torch.ge(IoU_max, 0.5)
        num_positive_anchors = positive_indices.sum()

        assigned_annotations = bbox_annotation[IoU_argmax, :]
        # targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
        alpha_factor = torch.ones(targets.shape) * alpha
        alpha_factor = alpha_factor.cuda()

        alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
        focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
        # focal_weight = torch.clamp(focal_weight, 1e-4, 1.0)
        focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

        bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

        cls_loss = focal_weight * bce

        cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
        classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

        # compute the loss for regression

        if positive_indices.sum() > 0:
            assigned_annotations = assigned_annotations[positive_indices, :]
            print('assigned_annotations.shape',assigned_annotations.shape)
            anchor_widths_pi = anchor_widths[positive_indices]
            anchor_heights_pi = anchor_heights[positive_indices]
            anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
            anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

            gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
            gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
            gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
            gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

            # clip widths to 1
            gt_widths = torch.clamp(gt_widths, min=1)
            gt_heights = torch.clamp(gt_heights, min=1)

            targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
            targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
            targets_dw = torch.log(gt_widths / anchor_widths_pi)
            targets_dh = torch.log(gt_heights / anchor_heights_pi)

            targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
            targets = targets.t()

            targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()

            negative_indices = ~positive_indices

            regression_diff = torch.abs(targets - regression[positive_indices, :])

            regression_loss = torch.where(
                torch.le(regression_diff, 1.0 / 9.0),
                0.5 * 9.0 * torch.pow(regression_diff, 2),
                regression_diff - 0.5 / 9.0
            )
            regression_losses.append(regression_loss.mean())
        else:
            regression_losses.append(torch.tensor(0).cuda().float())
        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
           torch.stack(regression_losses).mean(dim=0, keepdim=True)