"""训练全精度模型"""

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
from models.mobilenetv2 import mobilenetv2_quant
from utils import get_loader


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


def train_one_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    lossLayer = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = lossLayer(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()
            ))

def validate(model: nn.Module, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    lossLayer = torch.nn.CrossEntropyLoss(reduction='sum')
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += lossLayer(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        test_loss, 100. * correct / len(test_loader.dataset)
    ))


if __name__ == "__main__":
    random_seed(seed=42)
    args = _args()

    # parameters
    batch_size = 64
    test_batch_size = 64
    seed = 1
    epochs = 10
    lr = 0.01
    momentum = 0.5
    save_model_dir = 'examples/ckpt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    train_loader, test_loader = get_loader(args, batch_size)

    # choose model
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
    model = mobilenetv2_quant()

    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, device, train_loader, optimizer, epoch)
        validate(model, device, test_loader)
    
    if save_model_dir is not None:
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)

        model_save_path = os.path.join(save_model_dir, f'{args.dataset}_{model._get_name()}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f'model is saved to {model_save_path}')
