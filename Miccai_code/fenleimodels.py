
import torch
import torch.nn as nn
# import pretrainedmodels
from seresnext import se_resnext50_32x4d as pretrainedmodels_se_resnext50_32x4d
import re
from util import *
from torch.nn import functional as F


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class BinaryHead(nn.Module):
    def   __init__(self, num_class=2, emb_size=2048, s=16):
        super(BinaryHead, self).__init__()
        self.s = s
        self.fc = nn.Sequential(nn.Linear(emb_size, num_class))

    def forward(self, fea):
        fea = l2_norm(fea)
        logit = self.fc(fea) * self.s
        return logit

from collections import OrderedDict

def Se_resnext50_32x4d( pretrained, pth_path, **kwargs):
    model = se_resnext50_32x4d()
    epoch=0
    if pretrained:
        # state_dict = torch.load(pth_path)
        print('loadsuccess.....')
        epoch_counts = re.findall('\d+', pth_path)
        epoch = int(epoch_counts[len(epoch_counts) - 1])
        # model.load_state_dict(state_dict,strict=True)
        # load_model(model, pth_path)
        pretrained_dict=torch.load(pth_path)
        new_state_dict=OrderedDict()
        for k,v in pretrained_dict.items():
            name=k[7:]
            new_state_dict[name]=v
            model.load_state_dict(new_state_dict, strict=True)
        model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(pth_path).items()})
        # model.load_state_dict(torch.load(pth_path)["state_dict"])
    return model,epoch
class se_resnext50_32x4d(nn.Module):
    def __init__(self):
        super(se_resnext50_32x4d, self).__init__()

        self.model_ft = nn.Sequential(
            *list(pretrainedmodels_se_resnext50_32x4d(num_classes=1000, pretrained="imagenet").children())[
                :-2
            ]
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # print('222222',self.model_ft)
        self.model_ft.last_linear = None
        # print('111111',*list(pretrainedmodels_se_resnext50_32x4d(num_classes=1000, pretrained="imagenet").children())[
        #         :-2
        #     ])
        self.fea_bn = nn.BatchNorm1d(2048)
        self.fea_bn.bias.requires_grad_(False)
        self.binary_head = BinaryHead(5, emb_size=2048, s=1)
        self.dropout = nn.Dropout(p=0.2)


    def forward(self, x):
        with torch.no_grad():
            img_feature= self.model_ft(x)
            # output1=self.model_ft[0:1](x)
            # print('output1',output1.shape)
            # output1=self.model_ft[0:2][0](x)
            # print('output1',output1.shape)
            # output1=self.model_ft[0:2][:1](x)
            # print('output1',output1.shape)
            # output1=self.model_ft[0:2][:2](x)
            # print('output1',output1.shape)
            # output1=self.model_ft[0:2][:3](x)
            # print('output1',output1.shape)
            # output1=self.model_ft[0:2][:4](x)
            # print('output1',output1.shape)
            output1=self.model_ft[0:2](x)#[1,256,56,56]
            # print('output1',output1)
            print('output1',output1.shape)
            img_feature1 = self.avg_pool(output1).view(output1.size(0), -1)
            print('img_feature2',img_feature1.shape)
            img_feature1 =F.tanh(img_feature1)
            print('img_feature2',img_feature1.shape)
            output2=self.model_ft[0:3](x)
            # print('output2',output2)
            # print('output2',output2.shape)
            output3=self.model_ft[0:4](x)
            # print('output3',output3)
            # print('output3',output3.shape)
            output4=self.model_ft[0:5](x)
            # print('output4',output4)
            # print('output4',output4.shape)
            # output5=self.model_ft[0:6](x)
            # print('output5',output5)
            # output6=self.model_ft[0:7](x)
            # print('output6',output6)
            # output7=self.model_ft[0:8](x)
            # print('output7',output7)
            img_feature = self.avg_pool(img_feature)
            # print('img_feature1',img_feature.shape)
            img_feature = img_feature.view(img_feature.size(0), -1)
            fea = self.fea_bn(img_feature)
            # fea = self.dropout(fea)
            output = self.binary_head(fea)

        return output,img_feature1
        # return output
