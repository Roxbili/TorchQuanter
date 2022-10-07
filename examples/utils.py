import torch
from torchvision import datasets, transforms


def get_loader(args, batch_size):
    if args.dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.dataset_dir, train=True, download=True, 
                        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(args.mean, args.std)
                        ])),
            batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.dataset_dir, train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(args.mean, args.std)
                            ])),
            batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True
        )
    elif args.dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.dataset_dir, train=True, download=True, 
                        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(args.mean, args.std)
                        ])),
            batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.dataset_dir, train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(args.mean, args.std)
                            ])),
            batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True
        )
    else:
        raise ValueError(f"Unsupported dataset type {args.dataset}")
    return train_loader, test_loader

def export_onnx(args, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.dataset == 'mnist':
        dummy_input = torch.rand(1, 1, 28, 28).to(device)
    elif args.dataset == 'cifar10':
        dummy_input = torch.rand(1, 3, 32, 32).to(device)
    else:
        raise ValueError(f"Unsupported dataset type {args.dataset}")
    
    forward_bk = model.forward
    model.forward = model.quantize_inference
    torch.onnx.export(model, dummy_input, 
        'test.onnx', opset_version=11)
    model.forward = forward_bk
