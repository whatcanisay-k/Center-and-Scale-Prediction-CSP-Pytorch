from net.resnet import *
import torch
import  torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init

class Deconvolution(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super(Deconvolution, self).__init__()
        self.deconvolution = nn.ConvTranspose2d(in_channels, out_channels, 4, stride = stride, padding = padding)
        init.xavier_normal_(self.deconvolution.weight.data)

    def forward(self, x):
        x = self.deconvolution(x)
        return x

class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out



class Csp(nn.Module):
    def __init__(self, phase):
        super(Csp, self).__init__()
        self.phase = phase

        if self.phase == 'test':
            resnet = resnet50(pretrained=False, receptive_keep=True)
        else:
            resnet = resnet50(pretrained=True, receptive_keep=True)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.l2_norm_stage2 = L2Norm(256, 10)
        self.l2_norm_stage3 = L2Norm(512, 10)
        self.l2_norm_stage4 = L2Norm(1024, 10)
        self.l2_norm_stage5 = L2Norm(2048, 10)

        #self.deconv_stage2_1 = Deconvolution(256, 256, 4, 0)
        #self.deconv_stage2_2 = Deconvolution(768, 256, 4, 0)
        self.deconv_stage3 = Deconvolution(512, 256, 2, 1)
        self.deconv_stage4 = Deconvolution(1024, 256, 4, 0)
        self.deconv_stage5 = Deconvolution(2048, 256, 4, 0)

        # share feature map of center predict, scale regression, offsets regression
        self.conv_share_feature = nn.Conv2d(1024, 256, kernel_size = 3, stride = 1, padding = 1, bias = False)
        nn.init.xavier_normal_(self.conv_share_feature.weight)

        self.conv_center = nn.Conv2d(256, 2, kernel_size = 1, stride = 1)
        nn.init.xavier_normal_(self.conv_center.weight)

        self.conv_scale = nn.Conv2d(256, 2, kernel_size = 1, stride = 1)
        nn.init.xavier_normal_(self.conv_scale.weight)

        self.conv_offset = nn.Conv2d(256, 2, kernel_size = 1, stride = 1)
        nn.init.xavier_normal_(self.conv_offset.weight)


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x_stage2 = self.l2_norm_stage2(x)

        x = self.layer2(x)
        x_3 = self.l2_norm_stage3(x)
        x_3 = self.deconv_stage3(x_3)
        x_stage3 = self.relu(x_3)

        x = self.layer3(x)
        x_4 = self.l2_norm_stage4(x)
        x_4 = self.deconv_stage4(x_4)
        x_stage4 = self.relu(x_4)

        x = self.layer4(x)
        x_5 = self.l2_norm_stage5(x)
        x_5 = self.deconv_stage5(x_5)
        x_stage5 = self.relu(x_5)

        '''x_6 = torch.cat([x_stage3, x_stage4, x_stage5], dim=1)
        x_6 = self.deconv_stage2_2(x_6)
        x_6 = self.relu(x_6)
        x_7 = self.deconv_stage2_1(x_stage2)
        x_7 = self.relu(x_7)
        x = torch.cat([x_7, x_6], dim=1)'''

        x = torch.cat([x_stage2, x_stage3, x_stage4, x_stage5], dim=1)
        x = self.conv_share_feature(x)
        x = self.relu(x)

        center_cls_score = self.conv_center(x)
        scale_h_w = self.conv_scale(x)
        offset = self.conv_offset(x)

        if self.phase == 'test':
            center_cls_score = F.softmax(center_cls_score, 1)
            center_cls_score = center_cls_score[:, 1, :, :]

        return (center_cls_score, scale_h_w, offset)

