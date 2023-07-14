import torch.nn as nn
import torch
import math
import time
import torch.utils.model_zoo as model_zoo
from models.basic_block import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from models.anchors import Anchors
from torch.autograd import Variable
import cfg
from torchvision.ops import RoIPool
from torch.nn import functional as F
import numpy as np


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def py_cpu_nms(dets, thresh):
    dets = dets.cpu().numpy()
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the heigh  t of overlap

        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]  # because index start from 1

    return keep


def clipBoxes1(boxes):
    # batch_size, num_channels, height, width = img.shape
    height, width = cfg.patch_size

    boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
    boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

    boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
    boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)
    # print('--------------------------------clipbox')
    return boxes

def croppatch_model(classification, regression,anchors,inputs):

    bbox_id_all = []
    result_all = []

    for i0 in range(inputs.size(0)):
        classification1 =Variable(classification[i0].unsqueeze(0))
        regression1 = Variable(regression[i0].unsqueeze(0))
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        transformed_all = []
        bbox_id = []
        coord_id=[]
        # IOU_area=[]
        transformed_anchors = regressBoxes(anchors, regression1)
        transformed_anchors = clipBoxes1(transformed_anchors)
        scores = classification1
        scores_over_thresh = (scores > 0.05)[0, :, 0]
        transformed_anchors_1 = transformed_anchors[0, scores_over_thresh, :]
        classification2 = classification[i0, scores_over_thresh, :]
        d = torch.where(scores_over_thresh == True)
        scores = scores[0, scores_over_thresh, :]
        transformed_all.append(torch.cat([transformed_anchors_1, scores], dim=1))
        if bool(transformed_all):
            transformed_all = torch.cat(transformed_all, dim=0)
            if transformed_all.size(0) == 0:
                print('error')
            else:
                anchors_num_idx = py_cpu_nms(transformed_all, 0.05)
                # print('anchors_num_idx',anchors_num_idx)
                if len(anchors_num_idx) < cfg.patch_num:
                    print('error', len(anchors_num_idx))
                    patch_num = len(anchors_num_idx)
                else:
                    patch_num = cfg.patch_num

                for i in range(patch_num):
                # for i in range(len(anchors_num_idx)):
                    tx = int(transformed_anchors_1[anchors_num_idx[i], 0])
                    ty = int(transformed_anchors_1[anchors_num_idx[i], 1])
                    tx1 = int(transformed_anchors_1[anchors_num_idx[i], 2])
                    ty1 = int(transformed_anchors_1[anchors_num_idx[i], 3])
                    p=float(scores[anchors_num_idx[i], 0])
                    # print('=============copatch',transformed_all[anchors_num_idx[i]][4])

                    bbox_id.append(d[0][anchors_num_idx[i]])
                    size = 224
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

                    left_d = torch.tensor(left_d, dtype=torch.float).cuda()
                    up_d = torch.tensor(up_d, dtype=torch.float).cuda()
                    right_d = torch.tensor(right_d, dtype=torch.float).cuda()
                    low_d = torch.tensor(low_d, dtype=torch.float).cuda()
                    pos1 = [left_d, up_d, right_d, low_d]
                    coord_id.append(pos1)
        result_all.append(coord_id)
        bbox_id_all.append(bbox_id)
    return bbox_id_all,result_all

class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)
        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        Ce5 = out
        # print('Re1',out.shape)
        out = self.conv2(out)
        out = self.act2(out)
        # Ce5 = out
        # print('Re2',out.shape)
        out = self.conv3(out)
        out = self.act3(out)
        # print('Re3',out.shape)
        out = self.conv4(out)
        out = self.act4(out)
        # print('Re4',out.shape)
        out = self.output(out)
        # print('Re5',out.shape)
        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 4),Ce5


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):

        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.act4(out)
        Ce5=out
        out = self.output(out)
        out = self.output_act(out)
        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes),Ce5


class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers):
        ###50 LAYERS[3,3,6,3]
        ###50 LAYERS[3,4,6,3]
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        # self.focalLoss = losses.FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)
        self.roi1 = RoIPool((28, 28), 1/8)
        self.roi2 = RoIPool((14, 14), 1/16)
        self.roi3 = RoIPool((7, 7), 1/32)
        self.roi4 = RoIPool((4, 4), 1/64)
        self.roi5 = RoIPool((2, 2), 1/128)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs,mode):

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])
        CE_tensor=[]
        ROI_feature=[]
        for feature in features:
            CE_tensor.append(self.classificationModel(feature)[1])

        regression = torch.cat([self.regressionModel(feature)[0] for feature in features], dim=1)
        classification = torch.cat([self.classificationModel(feature)[0] for feature in features], dim=1)
        anchors = self.anchors(inputs).cuda()
        if mode=='train':
            bbox_id,patch_result= croppatch_model(classification, regression, anchors, inputs)
            for e,result in enumerate(patch_result):
                for e_id,coor in enumerate(result):
                    info = torch.zeros(1, 5).cuda()
                    info[0][1],info[0][2],info[0][3],info[0][4]=coor

                    if bbox_id[e][e_id]<147456:
                        pool=self.roi1(CE_tensor[0][e].unsqueeze(dim=0),info)
                        pool=self.avg_pool(pool).view(pool.size(0), -1)
                        pool = F.tanh(pool)

                    elif bbox_id[e][e_id]>=147456 and bbox_id[e][e_id]<184320:
                        pool=self.roi2(CE_tensor[1][e].unsqueeze(dim=0),info)
                        pool=self.avg_pool(pool).view(pool.size(0), -1)
                        pool = F.tanh(pool)
                    elif bbox_id[e][e_id]>=184320 and bbox_id[e][e_id]<193536:
                        pool=self.roi3(CE_tensor[2][e].unsqueeze(dim=0),info)
                        pool=self.avg_pool(pool).view(pool.size(0), -1)
                        pool = F.tanh(pool)
                    elif bbox_id[e][e_id] >= 193536 and bbox_id[e][e_id] < 195840:
                        pool = self.roi4(CE_tensor[3][e].unsqueeze(dim=0), info)
                        pool=self.avg_pool(pool).view(pool.size(0), -1)
                        pool = F.tanh(pool)
                    elif bbox_id[e][e_id] >= 195840 and bbox_id[e][e_id] < 196416:
                        pool = self.roi5(CE_tensor[4][e].unsqueeze(dim=0), info)
                        pool=self.avg_pool(pool).view(pool.size(0), -1)
                        pool = F.tanh(pool)
                    ROI_feature.append(pool)

            ROI_features=torch.cat(ROI_feature)


            return classification, regression, anchors,ROI_features
        elif mode=='eval':
            return classification, regression, anchors

def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    for param in model.parameters():
        param.requires_grad = False
    # for param in model.classificationModel.conv3.parameters():
    #     param.requires_grad = True
    # for param in model.fpn.parameters():
    #     param.requires_grad = True
    for param in model.classificationModel.parameters():
        param.requires_grad = True
    # for param in model.classificationModel.parameters():
    #     param.requires_grad = True
    for param in model.regressionModel.parameters():
        param.requires_grad = True
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model


if __name__ == "__main__":
    retinanet = resnet50(num_classes=2, pretrained=False)
    retinanet.eval()
    input = torch.randint(0, 255, [2, 3, 256, 256]).float()
    cls, reg, anchors = retinanet(input)
    print(cls.shape, reg.shape, anchors.shape)
