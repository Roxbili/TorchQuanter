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
