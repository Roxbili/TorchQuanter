"""后训练量化"""

import os, sys
sys.path.append(os.path.join(os.getcwd(), 'examples/'))
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from models.model import (
    Model, ModelBN, ModelLinear, ModelShortCut, ModelBNNoReLU,
    ModelLayerNorm, ModelAttention, ModelMV2, ModelMV2Naive, ModelDepthwise,
    ModelMV2ShortCut, ModelTransformerEncoder, ModelConvEncoder
)
from torchquanter.utils import random_seed

def full_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Full Model Accuracy: {:.2f}%\n'.format(100. * correct / len(test_loader.dataset)))

def quantize(model: Model, loader):
    for i, (data, target) in enumerate(loader, 1):
        data, target = data.to(device), target.to(device)
        output = model.quantize_forward(data)
    print('quantization finish')

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

    # parameters
    batch_size = 64
    save_model_dir = 'examples/ckpt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    if os.path.exists('/share/Documents/project/dataset'):
        dataset_path = '/share/Documents/project/dataset/mnist'
    elif os.path.exists('/home/LAB/leifd/dataset'):
        dataset_path = '/home/LAB/leifd/dataset/mnist'
    else:
        raise Exception

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(dataset_path, train=False, download=False,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True
    )

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
    model = ModelConvEncoder()

    model = model.to(device)
    state_dict = torch.load(os.path.join(save_model_dir, f'mnist_{model._get_name()}.pth'), map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    full_inference(model, test_loader)  # 测试模型全精度的精度

    # 量化
    num_bits = 8
    print('Quantization bit: %d' % num_bits)
    model.quantize(num_bits=num_bits, signed=True)
    model.eval()
    quantize(model, test_loader)
    model.freeze()

    # 量化推理
    quantize_inference(model, test_loader)