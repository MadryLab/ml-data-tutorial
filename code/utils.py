"""
This file contains utility functions for the code snippets in Section 3 of the tutorial.
"""

import numpy as np
import torch
import torchvision

from matplotlib import pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler


# Resnet9 architecture
class Mul(torch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


def get_model(num_classes=10):
    def conv_bn(
        channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1
    ):
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                channels_in,
                channels_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            torch.nn.BatchNorm2d(channels_out),
            torch.nn.ReLU(inplace=True),
        )

    model = torch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(torch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),
        Residual(torch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        torch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        torch.nn.Linear(128, num_classes, bias=False),
        Mul(0.2),
    )
    return model.cuda()


def train(
    model,
    loader,
    lr=0.4,
    epochs=24,
    momentum=0.9,
    weight_decay=5e-4,
    lr_peak_epoch=5,
    label_smoothing=0.0,
    model_id=0,
):

    opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    iters_per_epoch = len(loader)
    # Cyclic LR with single triangle
    lr_schedule = np.interp(
        np.arange((epochs + 1) * iters_per_epoch),
        [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
        [0, 1, 0],
    )
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)

    for ep in range(epochs):
        for ims, labs in loader:
            ims = ims.cuda()
            labs = labs.cuda()
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(ims)
                loss = loss_fn(out, labs)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

    return model


def train_on_subset(subset):
    # train a model on the subset and return it
    model = get_model()
    loader = get_loader(split="train", shuffle=True, indices=subset)
    return train(model, loader)


def record_outputs(model, target_examples):
    # record the margin of the model on the target example
    images, labels = target_examples
    images = images.cuda()
    labels = labels.cuda()
    logits = model(images)
    bindex = torch.arange(logits.shape[0]).to(logits.device, non_blocking=False)
    logits_correct = logits[bindex, labels]

    cloned_logits = logits.clone()
    # remove the logits of the correct labels from the sum
    # in logsumexp by setting to -ch.inf
    cloned_logits[bindex, labels] = torch.tensor(
        -torch.inf, device=logits.device, dtype=logits.dtype
    )

    margins = logits_correct - cloned_logits.logsumexp(dim=-1)
    return margins.clone().detach().cpu()


def get_loader(
    split="train",
    batch_size=256,
    num_workers=8,
    shuffle=False,
    augment=True,
    indices=None,
):
    if augment:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomAffine(0),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201)
                ),
            ]
        )
    else:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201)
                ),
            ]
        )

    is_train = split == "train"
    dataset = torchvision.datasets.CIFAR10(
        root="/tmp/cifar/", download=True, train=is_train, transform=transforms
    )
    # we are only going to use 5000 training samples from the CIFAR-10 training set
    # 2500 from the "cat" class, and 2500 from the "dog" class
    if is_train:
        cats = np.where(np.array(dataset.targets) == 3)[0][:2500]
        dogs = np.where(np.array(dataset.targets) == 5)[0][:2500]
        inds = np.concatenate([cats, dogs])

        # further subsetting (e.g., used when computing the linear datamodeling score)
        if indices is not None:
            inds = inds[indices]

        dataset = torch.utils.data.Subset(dataset, sorted(inds))

    loader = torch.utils.data.DataLoader(
        dataset=dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers
    )

    return loader


def plot_images(tensors, title=None):
    if len(tensors) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(1, 1))
        ax.axis("off")
        image = tensors[0].permute(1, 2, 0).cpu().numpy()
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        ax.imshow(image)
        fig.show()
    else:
        fig, axes = plt.subplots(1, len(tensors), figsize=(1, len(tensors)))
        fig.suptitle(title, fontsize=11)
        for ax, tensor in zip(axes, tensors):
            ax.axis("off")
            image = tensor.permute(1, 2, 0).cpu().numpy()
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
            ax.imshow(image)
        fig.show()
