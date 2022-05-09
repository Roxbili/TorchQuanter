"""训练全精度模型"""

import os, sys
sys.path.append(os.path.join(os.getcwd(), 'examples/'))
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from models.model import Model, ModelBN, ModelLinear, ModelShortCut

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
    # parameters
    batch_size = 64
    test_batch_size = 64
    seed = 1
    epochs = 2
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

    # choose model
    # model = Model()
    # model = ModelBN()
    # model = ModelLinear()
    model = ModelShortCut()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, device, train_loader, optimizer, epoch)
        validate(model, device, test_loader)
    
    if save_model_dir is not None:
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)

        model_save_path = os.path.join(save_model_dir, f'mnist_{model._get_name()}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f'model is saved to {model_save_path}')