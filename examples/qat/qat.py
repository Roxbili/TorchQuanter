import os, sys

sys.path.append(os.path.join(os.getcwd(), 'examples/'))
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from models.model import Model, ModelBN, ModelLinear, ModelShortCut, ModelLayerNorm, ModelAttention, ModelBNNoReLU, ModelMV2
from torchquanter.utils import random_seed


def quantize_aware_training(model: Model, device, train_loader, optimizer, epoch):
    lossLayer = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model.quantize_forward(data)
        loss = lossLayer(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print('Quantize Aware Training Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()
            ))


def full_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Full Model Accuracy: {:.2f}%\n'.format(100. * correct / len(test_loader.dataset)))


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

    batch_size = 64
    epochs = 3
    lr = 0.01
    momentum = 0.5
    save_model_dir = 'examples/ckpt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/share/Documents/project/dataset/mnist', train=True, download=True, 
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/share/Documents/project/dataset/mnist', train=False, download=False,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True
    )

    # 加载训练好的全精度模型
    # model = Model()
    # model = ModelBN()
    # model = ModelBNNoReLU()
    # model = ModelLinear()
    # model = ModelShortCut()
    # model = ModelLayerNorm()
    # model = ModelAttention()
    model = ModelMV2()

    state_dict = torch.load(os.path.join(save_model_dir, f'mnist_{model._get_name()}.pth'), map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    full_inference(model, test_loader)  # 测试模型全精度的精度

    # init
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    num_bits = 8
    model.quantize(num_bits=num_bits, signed=True)
    print('Quantization bit: %d' % num_bits)

    # train
    model.train()
    for epoch in range(1, epochs + 1):
        quantize_aware_training(model, device, train_loader, optimizer, epoch)

    # save qat model
    model.eval()
    torch.save(model.state_dict(), os.path.join(save_model_dir, f'mnist_{model._get_name()}_qat.pth'))

    # fp32 -> int8/uint8
    model.freeze()

    # 量化推理
    quantize_inference(model, test_loader)
