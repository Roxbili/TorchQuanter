'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init
from torchquanter.nn import *


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class Feature(nn.Module):
    def __init__(self, cfg, batch_norm=False):
        super(Feature, self).__init__()
        self.cfg = cfg
        self.batch_norm = batch_norm
        layers = []
        in_channels = 3
        for v in self.cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def quantize(self, first_qi=True, num_bits=8, signed=True):
        qlayers = []
        i = 0
        for v in self.cfg:
            if v == 'M':
                qlayers += [QMaxPool2d(
                    self.layers[i], qi=False,
                    num_bits=num_bits, signed=signed)]
                i += 1
            elif self.batch_norm:
                qlayers += [QConvBNReLU(
                    self.layers[i], self.layers[i + 1], 
                    relu=True, qi=first_qi, num_bits=num_bits, signed=True)]
                i += 3
                first_qi = False
            else:
                qlayers += [QConv2d(
                    self.layers[i], relu=True, qi=first_qi,
                    num_bits=num_bits, signed=True)]
                i += 2 
                first_qi = False
        self.qlayers = nn.Sequential(*qlayers)

    def quantize_forward(self, x):
        return self.qlayers(x)

    def freeze(self, qi=None):
        for op in self.qlayers:
            qi = op.freeze(qi=qi)
        return qi   # return last op qo

    def quantize_inference(self, qx, mode='cmsis_nn'):
        for op in self.qlayers:
            qx = op.quantize_inference(qx, mode=mode)
        return qx


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features: Feature):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def quantize(self, num_bits=8, signed=True):
        self.features.quantize(first_qi=True, num_bits=num_bits, signed=signed)
        self.qclassifier = nn.Sequential(
            nn.Dropout(),
            QLinear(self.classifier[1], relu=True, qi=False, num_bits=num_bits, signed=signed),
            nn.Dropout(),
            QLinear(self.classifier[4], relu=True, qi=False, num_bits=num_bits, signed=signed),
            QLinear(self.classifier[6], relu=False, qi=False, num_bits=num_bits, signed=signed),
        )

    def quantize_forward(self, x):    
        x = self.features.quantize_forward(x)
        x = x.view(x.size(0), -1)
        x = self.qclassifier(x)
        return x

    def freeze(self):
        qo = self.features.freeze()
        for op in self.qclassifier:
            if not isinstance(op, nn.Dropout):
                qo = op.freeze(qi=qo)
        return qo

    def quantize_inference(self, x, mode='cmsis_nn'):
        qx = self.features.qlayers[0].qi.quantize_tensor(x)
        qx = self.features.quantize_inference(qx, mode=mode)
        qx = qx.view(qx.size(0), -1)
        for op in self.qclassifier:
            if not isinstance(op, nn.Dropout):
                qx = op.quantize_inference(qx, mode=mode)
        x = self.qclassifier[-1].qo.dequantize_tensor(qx)
        return x


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")"""
    return VGG(Feature(cfg['A']))


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(Feature(cfg['A'], batch_norm=True))


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")"""
    return VGG(Feature(cfg['B']))


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(Feature(cfg['B'], batch_norm=True))


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")"""
    return VGG(Feature(cfg['D']))


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(Feature(cfg['D'], batch_norm=True))


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")"""
    return VGG(Feature(cfg['E']))


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(Feature(cfg['E'], batch_norm=True))
