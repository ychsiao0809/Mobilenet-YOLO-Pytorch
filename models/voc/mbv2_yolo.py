from collections import OrderedDict

import torch
import torch.nn as nn
from models.voc.mobilenetv2 import mobilenetv2
from models.voc.yolo_loss import *
from torch.nn import init
import yaml
from utils.box import nms
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,depthwise=False):
        super(BasicConv, self).__init__()
        if depthwise == False :
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        else :
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False,groups = in_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)   
          
class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            #BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x
class PartAdd(nn.Module):
    def __init__(self):
        super(PartAdd, self).__init__()
        self.ff = nn.quantized.FloatFunctional()
    
    def forward(self, x, y):
        if x.size(1) == y.size(1):
            return self.ff.add(x,y)
        len = min(x.size(1),y.size(1))
#        new_1 = x[:,:len,...] + y[:,:len,...]
        new_1 = self.ff.add(x[:,:len,...], y[:,:len,...])
        if y.size(1) > x.size(1):
            new_2 = y[:,len:,...]
        else:
            new_2 = x[:,len:,...]
        new = torch.cat((new_1,new_2),1)
    
        return new
def DepthwiseConvolution(in_filters,out_filters):
    m = nn.Sequential(
        BasicConv(in_filters, in_filters, 3,depthwise=True),
        BasicConv(in_filters, in_filters, 1),
        BasicConv(in_filters, out_filters, 1 ),
    )
    return m
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        BasicConv(in_filters, in_filters, 3,depthwise=True),
        BasicConv(in_filters, in_filters, 1),
        BasicConv(in_filters, filters_list[0], 1),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m
class Connect(nn.Module):
    def __init__(self, channels):
        super(Connect, self).__init__()

        self.conv = nn.Sequential(
            BasicConv(channels, channels, 3,depthwise=True),
            BasicConv(channels, channels, 1 ),
        )
        self.ff = nn.quantized.FloatFunctional()
    def forward(self, x,):        
        x2 = self.conv(x)
#        x = torch.add(x,x2)
        x = self.ff.add(x,x2)
        return x
class yolo(nn.Module):
    def __init__(self,config):
        super(yolo, self).__init__()
        self.num_classes = config["yolo"]["num_classes"]
        self.num_anchors = config["yolo"]["num_anchors"]
        #  backbone
        model_url = 'https://raw.githubusercontent.com/d-li14/mobilenetv2.pytorch/master/pretrained/mobilenetv2-c5e733a8.pth'
        self.backbone = mobilenetv2(model_url)

        self.conv_for_S32 = BasicConv(1280,512,1)
        #print(num_anchors * (5 + num_classes))
        self.connect_for_S32 = Connect(512)
        self.yolo_headS32 = yolo_head([1024, self.num_anchors * (5 + self.num_classes)],512)
        
        
        self.upsample = Upsample()
        self.conv_for_S16 = DepthwiseConvolution(96,256)
        self.connect_for_S16 = Connect(256)
        self.yolo_headS16 = yolo_head([512, self.num_anchors * (5 + self.num_classes)],512)
        self.yolo_losses = []
        for i in range(2):
            self.yolo_losses.append(YOLOLoss(config["yolo"]["anchors"],config["yolo"]["mask"][i] \
                ,self.num_classes,[config["img_w"],config["img_h"]],config["yolo"]["ignore_thresh"][i],config["yolo"]["iou_thresh"],iou_weighting=config["iou_weighting"]))
        self.part_add = PartAdd()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
       
    def forward(self, x, targets=None):

        for i in range(2):
            self.yolo_losses[i].img_size = [x.size(2),x.size(3)]
        feature1, feature2 = self.backbone(x)
        S32 = self.conv_for_S32(feature2)
        S32 = self.connect_for_S32(S32)
        out0 = self.yolo_headS32(S32) 
        S32_Upsample = self.upsample(S32)
        S16 = self.conv_for_S16(feature1)
        S16 = self.connect_for_S16(S16)
        #S16 = self.blending(S16,S32_Upsample)
        #S16 = PartAdd(S16,S32_Upsample)
        S16 = self.part_add(S16,S32_Upsample)
        #print(S16.shape)
        #S16 = torch.add(S16,S32_Upsample)
       
        out1 = self.yolo_headS16(S16)
        out0 = self.dequant(out0)
        out1 = self.dequant(out1)

        output = self.yolo_losses[0](out0,targets),self.yolo_losses[1](out1,targets)
        if targets == None :
            output = nms(output,self.num_classes)

        output = [self.quant(out) for out in output]

        return output

class YoloWithoutLoss(nn.Module):
    def __init__(self,config):
        super(YoloWithoutLoss, self).__init__()
        self.num_classes = config["yolo"]["num_classes"]
        self.num_anchors = config["yolo"]["num_anchors"]
        #  backbone
        model_url = 'https://raw.githubusercontent.com/d-li14/mobilenetv2.pytorch/master/pretrained/mobilenetv2-c5e733a8.pth'
        self.backbone = mobilenetv2(model_url)

        self.conv_for_S32 = BasicConv(1280,512,1)
        #print(num_anchors * (5 + num_classes))
        self.connect_for_S32 = Connect(512)
        self.yolo_headS32 = yolo_head([1024, self.num_anchors * (5 + self.num_classes)],512)


        self.upsample = Upsample()
        self.conv_for_S16 = DepthwiseConvolution(96,256)
        self.connect_for_S16 = Connect(256)
        self.yolo_headS16 = yolo_head([512, self.num_anchors * (5 + self.num_classes)],512)
        self.part_add = PartAdd()

    def forward(self, x):

#        for i in range(2):
#            self.yolo_losses[i].img_size = [x.size(2),x.size(3)]
        feature1, feature2 = self.backbone(x)
        S32 = self.conv_for_S32(feature2)
        S32 = self.connect_for_S32(S32)
        out0 = self.yolo_headS32(S32)
        S32_Upsample = self.upsample(S32)
        S16 = self.conv_for_S16(feature1)
        S16 = self.connect_for_S16(S16)
        #S16 = self.blending(S16,S32_Upsample)
        #S16 = PartAdd(S16,S32_Upsample)
        S16 = self.part_add(S16,S32_Upsample)
        #print(S16.shape)
        #S16 = torch.add(S16,S32_Upsample)

        out1 = self.yolo_headS16(S16)

        return (out0, out1)

class YoloLoss(nn.Module):
    def __init__(self, config):
        super(YoloLoss, self).__init__()
        self.num_classes = config["yolo"]["num_classes"]
        self.yolo_losses = []
        for i in range(2):
            self.yolo_losses.append(YOLOLoss(config["yolo"]["anchors"],config["yolo"]["mask"][i] \
                ,self.num_classes,[config["img_w"],config["img_h"]],config["yolo"]["ignore_thresh"][i],config["yolo"]["iou_thresh"],iou_weighting=config["iou_weighting"]))

    def forward(self, out0, out1, targets, h, w):
        for i in range(2):
            self.yolo_losses[i].img_size = [h,w]

        output = self.yolo_losses[0](out0,targets),self.yolo_losses[1](out1,targets)
        if targets == None :
            output = nms(output,self.num_classes)

        return output
    
#def test():
#    net = yolo(3,20)
#    print(net)

#test()
