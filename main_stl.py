import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor
from tqdm.auto import trange

from LOSS import MTLLoss, MTLLoss2
from models import StlNet
from training import train_stl, validation_stl


def main():
    train_set = CIFAR10(".", train=True, download=True, transform=ToTensor(),
                        target_transform=lambda y: torch.tensor(y).long())
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
    val_set = CIFAR10(".", train=False, download=True, transform=ToTensor(),
                      target_transform=lambda y: torch.tensor(y).long())
    val_loader = DataLoader(val_set, batch_size=512, shuffle=True)
    epochs = 100
    ws = np.zeros([epochs, 2])
    model = StlNet(train_set[0][0].shape).cuda()
    criterion = MTLLoss2()
    params = list(model.parameters()) + list(criterion.parameters())

    # print(len(criterion.log_vars.T))
    # for i in range(len(criterion.log_vars.T)):
    #     print(criterion.log_vars[0, i])
    #     params += [criterion.log_vars[0, i]]
    #     criterion.log_vars[0, i].requires_grad = True

    optimizer = AdamW(params, lr=0.001)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    losses = []
    loop = trange(epochs)
    for epoch in loop:
        train_loss, train_acc = train_stl(model, data_loader=train_loader, optimizer=optimizer)
        val_loss, val_acc = validation_stl(model, val_loader)

        loop.set_postfix({"train clf": train_loss,
                          "val clf": val_loss,
                          "train acc": train_acc, "val acc": val_acc})

        scheduler.step()


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
