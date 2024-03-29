"""tflite后训练量化"""

import os, sys
sys.path.append(os.path.join(os.getcwd(), 'examples/'))
import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from models.model import (
    Model, ModelBN, ModelLinear, ModelShortCut, ModelBNNoReLU,
    ModelLayerNorm, ModelAttention, ModelMV2, ModelMV2Naive, ModelDepthwise,
    ModelMV2ShortCut, ModelTransformerEncoder, ModelConvEncoder,
    TinyFormerSupernetDMTPOnePath
)
from torchquanter.utils import random_seed
from converter import Torch2TFLiteConverter


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', metavar='DIR', default='/tmp',
                    help='path to dataset')
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

def representative_dataset():
    for data, label in test_loader.dataset:
        data = data.unsqueeze(0)
        yield [data]

if __name__ == "__main__":
    random_seed(seed=42)
    args = _args()

    # parameters
    batch_size = 64
    save_model_dir = 'examples/ckpt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.dataset_dir, train=False, download=False,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True
    )

    # 加载模型
    # model = Model()
    # model = ModelBN()
    model = ModelBNNoReLU()
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

    model = model.to(device)
    state_dict = torch.load(os.path.join(save_model_dir, f'mnist_{model._get_name()}.pth'), map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    full_inference(model, test_loader)  # 测试模型全精度的精度

    # 量化
    tmp_path = '/tmp/model.pth'
    torch.save(model, tmp_path)
    converter = Torch2TFLiteConverter(
        torch_model_path=tmp_path,
        tf_model_path='/tmp/model_converter/tf_model',
        tflite_model_save_path=os.path.join(save_model_dir, f'mnist_{model._get_name()}.lite'),
        target_shape=(28,28,1),
        representative_dataset=representative_dataset,
        evaluate_loader=test_loader
    )
    converter.convert()