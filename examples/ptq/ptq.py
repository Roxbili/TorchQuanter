"""后训练量化"""

import os, sys
sys.path.append(os.path.join(os.getcwd(), 'examples/'))
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from models.model import (
    Model, ModelBN, ModelLinear, ModelShortCut, ModelBNNoReLU,
    ModelLayerNorm, ModelAttention, ModelMV2, ModelMV2Naive, ModelDepthwise,
    ModelMV2ShortCut, ModelTransformerEncoder, ModelConvEncoder,
    TinyFormerSupernetDMTPOnePath
)
from torchquanter.utils import random_seed
from models.resnet import resnet18_quant
from utils import get_loader, export_onnx


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist', help='Type of dataset')
    parser.add_argument('--dataset-dir', metavar='DIR', default='/tmp',
                    help='Path to dataset')
    parser.add_argument('--mean', type=float, nargs='+', default=[0.1307,], metavar='MEAN',
                    help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float, nargs='+', default=[0.3081,], metavar='STD',
                        help='Override std deviation of of dataset')
    args = parser.parse_args()
    return args


def full_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Full Model Accuracy: {:.2f}%\n'.format(100. * correct / len(test_loader.dataset)))

def quantize(model: Model, loader):
    correct = 0
    for i, (data, target) in enumerate(loader, 1):
        data, target = data.to(device), target.to(device)
        output = model.quantize_forward(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('quantization finish')
    print('Train set: quantize_forward Accuracy: {:.2f}%\n'.format(100. * correct / len(loader.dataset)))

def quantize_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        data, target = data.to(device), target.to(device)
        output = model.quantize_inference(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Quant Model Accuracy: {:.2f}%\n'.format(100. * correct / len(test_loader.dataset)))

if __name__ == "__main__":
    random_seed(seed=42)
    args = _args()

    # parameters
    batch_size = 64
    save_model_dir = 'examples/ckpt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    train_loader, test_loader = get_loader(args, batch_size)

    # 加载模型
    # model = Model()
    # model = ModelBN()
    # model = ModelBNNoReLU()
    # model = ModelLinear()
    # model = ModelShortCut()
    # model = ModelLayerNorm()
    # model = ModelAttention()
    # model = ModelDepthwise()
    # model = ModelMV2Naive()
    # model = ModelMV2()
    # model = ModelMV2ShortCut()
    # model = ModelTransformerEncoder()
    # model = ModelConvEncoder()
    # model = TinyFormerSupernetDMTPOnePath(
    #     num_classes=10, downsample_layers=1, mv2block_layers=1,
    #     transformer_layers=1, channel=[8, 8, 8], last_channel=8,
    #     transformer0_embedding_dim=[16], transformer0_dim_feedforward=[16],
    #     transformer1_embedding_dim=[16], transformer1_dim_feedforward=[16],
    #     choice=[1,0,0,0], first_channel=1
    # )
    model = resnet18_quant()

    model = model.to(device)
    state_dict = torch.load(os.path.join(save_model_dir, f'{args.dataset}_{model._get_name()}.pth'), map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    full_inference(model, test_loader)  # 测试模型全精度的精度

    # 量化
    num_bits = 8
    print('Quantization bit: %d' % num_bits)
    model.quantize(num_bits=num_bits, signed=True, symmetric_feature=True)
    model = model.to(device)
    model.eval()
    quantize(model, train_loader)
    model.freeze()

    # 量化推理
    quantize_inference(model, test_loader)

    # 保存参数
    save_path = os.path.join(save_model_dir, f'{args.dataset}_{model._get_name()}_ptq.pth')
    torch.save(model.state_dict(), save_path)

    # 加载参数
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict)

    # 量化推理
    quantize_inference(model, test_loader)

    try:
        export_onnx(args, model)
    except torch.onnx.CheckerError:
        pass
    except Exception as e:
        raise e
