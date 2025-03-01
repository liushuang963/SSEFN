import torch
from torch import nn
from torchvision import models
from sklearn.decomposition import PCA
import torch.nn.functional as F

middle_dim = 256

class ConvModule(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size=3,pad_size=1,stride=1,dilation=1,dropout=0, use_residual=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=pad_size,dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.use_residual = use_residual
        if use_residual:
            self.res1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=stride,padding=pad_size,dilation=dilation),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU()
                                      )
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=pad_size,dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activate2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)



    def forward(self,x):
        if self.use_residual:
            x_org = x
        x = self.conv(x)
        x = self.dropout1(x)
        x = self.bn(x)
        x = self.activate(x)
        if self.use_residual:
            x = x + self.res1(x_org)
            x_org_2 = x

        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.bn2(x)
        x = self.activate2(x)
        if self.use_residual:
            x = x + x_org_2
        return x

class Spatial3x3(nn.Module):
    def __init__(self, in_channels, num_classes,dropout, fusion_mode='decision'):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.conv1 = ConvModule(in_channels=in_channels,out_channels=middle_dim,kernel_size=3,pad_size=1,stride=1,dropout=dropout)
        self.pool1 = nn.AvgPool2d(kernel_size=3,stride=3)
        if self.fusion_mode=="decision":
            self.classifier = nn.Linear(middle_dim, num_classes)
        self.att = nn.Linear(middle_dim, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = x.view(x.size(0),x.size(1))
        att = nn.Sigmoid()(self.att(x))
        if self.fusion_mode=="decision":
            logits = self.classifier(x)
            return logits, att
        else:
            return x, att


class Spatial9x9(nn.Module):
    def __init__(self,in_channels, num_classes,dropout, fusion_mode='decision'):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.conv1 = ConvModule(in_channels=in_channels, out_channels=middle_dim, kernel_size=3, pad_size=1, stride=1,dropout=dropout)
        # self.pool1 = nn.AvgPool2d(kernel_size=3, stride=3)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=3)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        # self.conv2 = ConvModule(in_channels=middle_dim, out_channels=512,dropout=dropout)
        # self.pool2 = nn.AvgPool2d(kernel_size=3, stride=3)
        if self.fusion_mode=="decision":
            self.classifier = nn.Linear(middle_dim, num_classes)
        self.att = nn.Linear(middle_dim, 1)
        #     self.classifier = nn.Linear(512, num_classes)
        # self.att = nn.Linear(512, 1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.gap(x)
        # x = self.conv2(x)
        # x = self.pool2(x)
        x = x.view(x.size(0), x.size(1))

        att = nn.Sigmoid()(self.att(x))
        if self.fusion_mode=="decision":
            logits = self.classifier(x)
            return logits, att
        else:
            return x, att


class Spatial27x27(nn.Module):
    def __init__(self, in_channels, num_classes,dropout, fusion_mode='decision'):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.conv1 = ConvModule(in_channels=in_channels, out_channels=middle_dim, kernel_size=3, pad_size=1, stride=1,dropout=dropout)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=3)
        # self.conv2 = ConvModule(in_channels=middle_dim, out_channels=512,dropout=dropout)
        # self.pool2 = nn.AvgPool2d(kernel_size=3, stride=3)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        # self.classifier = nn.Linear(512, num_classes)
        # self.att = nn.Linear(512, 1)

        if self.fusion_mode=="decision":
            self.classifier = nn.Linear(middle_dim, num_classes)
        self.att = nn.Linear(middle_dim, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        # x = self.conv2(x)
        # x = self.pool2(x)
        x = self.gap(x)
        x = x.view(x.size(0), x.size(1))

        att = nn.Sigmoid()(self.att(x))
        logits = self.classifier(x)
        return logits, att



class Spatial1x1(nn.Module):
    def __init__(self, in_channels, num_classes,dropout=0.2, fusion_mode='decision'):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.proj = nn.Sequential(nn.Linear(in_channels,middle_dim),
                                  nn.Dropout(dropout),
                                  nn.BatchNorm1d(middle_dim),
                                  nn.ReLU(),
                                  # nn.Linear(middle_dim, 512),
                                  nn.Linear(middle_dim, middle_dim),
                                  nn.Dropout(dropout),
                                  # nn.BatchNorm1d(512),
                                  nn.BatchNorm1d(middle_dim),
                                  nn.ReLU()
                                  )

        if self.fusion_mode=="decision":
            self.classifier = nn.Linear(middle_dim, num_classes)
        self.att = nn.Linear(middle_dim, 1)

    def forward(self,x):
        feature = self.proj(x)
        att = nn.Sigmoid()(self.att(feature))
        if self.fusion_mode=="decision":
            logits = self.classifier(feature)
            return logits, att
        else:
            return feature, att


class MSBranch(nn.Module):
    # def __init__(self, in_channels, num_classes, mode=['1','3','9','27'], use_att=True,dropout=0.2):
    def __init__(self, in_channels, num_classes, mode=['27'], use_att=True,dropout=0, fusion_mode='decision'):
        super().__init__()
        self.mode = mode
        self.use_att = use_att
        self.mode = mode
        self.fusion_mode = fusion_mode
        module_list = []
        for mode_id in mode:
            if mode_id == "1":
                module = Spatial1x1(in_channels,num_classes,dropout, fusion_mode)
            elif mode_id == "3":
                module = Spatial3x3(in_channels, num_classes, dropout, fusion_mode)
            elif mode_id == "9":
                module = Spatial9x9(in_channels, num_classes, dropout, fusion_mode)
            elif mode_id == "27":
                module = Spatial27x27(in_channels, num_classes, dropout, fusion_mode)
            module_list.append(module)
        self.module_list = nn.ModuleList(module_list)


    def forward(self, x):
        patch_size = x.size()[2]
        patch_length = int((patch_size - 1)/2)
        center_pos = patch_length
        x_1x1 = x[:,:,center_pos,center_pos]
        x_3x3 = x[:,:,center_pos-1:center_pos+1 + 1,center_pos-1:center_pos+1 + 1]
        x_9x9 = x[:,:,center_pos-4:center_pos+4 + 1,center_pos-4:center_pos+4 + 1]
        x_27x27 = x[:, :, center_pos - 13:center_pos + 13 + 1, center_pos - 13:center_pos + 13 + 1]
        i = 0
        sum_logits = 0
        logits_dict = {}
        for i, module_layer in enumerate(self.module_list):
            scale = self.mode[i]
            if scale == '1':
                input = x_1x1
            elif scale == '3':
                input = x_3x3
            elif scale == '9':
                input = x_9x9
            elif scale == '27':
                input = x_27x27
            logits_i, att_i = module_layer(input)
            logits_dict["{}_{}".format(i,scale)] = logits_i
            if self.use_att:
                sum_logits = sum_logits + logits_i * att_i
            else:
                sum_logits = sum_logits + logits_i

        if self.fusion_mode=="decision":
            mean_logits = sum_logits / (i+1)
            return mean_logits,logits_dict
        else:
            return sum_logits,logits_dict


class SSEFN(nn.Module):

    def __init__(self, in_channels, num_classes, spe_mode=['spatial', '1', '2', '4', '8'],
                 spatial_mode=['1', '3', '9', '27'], use_att=True, dropout=0.2, fusion_mode='decision'):
        super(SSEFN, self).__init__()
        self.spe_mode = spe_mode
        self.fusion_mode=fusion_mode
        spe_list = []
        for spe_id in spe_mode:
            if 'spatial' == spe_id:
                i = 0
            else:
                i = int(spe_id)
            spe = MSBranch(in_channels-i, num_classes, mode=spatial_mode, use_att=use_att, dropout=dropout, fusion_mode=fusion_mode)
            spe_list.append(spe)
        self.spe_list = nn.ModuleList(spe_list)


    def forward(self,x, train=False,return_dict=False):
        x = x.permute(0, 3, 1, 2)
        sum_logits = 0
        logits_dict = {}
        for i, spe_layer in enumerate(self.spe_list):
            spe_id = self.spe_mode[i]
            if spe_id == "spatial":
                j = 0
                x_j = x
            else:
                j = int(spe_id)
                x_j = x[:, j:, :, :] - x[:, :-j, :, :]

            mean_logits_spe_j, spe_j_dict = spe_layer(x_j)
            for key in spe_j_dict.keys():
                logits_dict['spe_{}_'.format(spe_id) + key] = spe_j_dict[key]
            sum_logits = mean_logits_spe_j + sum_logits


        sum_logits = sum_logits / (i+1)

        if self.training or return_dict:
            return sum_logits, logits_dict
        else:
            return sum_logits
