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
from optimization import PCGrad
from LOSS import MTLUW, MTLUWR
from models import Net
from training import train, validation, train_pc


def main():
    train_set = CIFAR10(".", train=True, download=True, transform=ToTensor(),
                        target_transform=lambda y: torch.tensor(y).long())
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
    val_set = CIFAR10(".", train=False, download=True, transform=ToTensor(),
                      target_transform=lambda y: torch.tensor(y).long())
    val_loader = DataLoader(val_set, batch_size=512, shuffle=True)
    epochs = 30

    model = Net(in_shape=train_set[0][0].shape).cuda()
    criterion = MTLUWR()
    params = list(model.parameters())

    # print(len(criterion.log_vars.T))
    # for i in range(len(criterion.log_vars.T)):
    #     print(criterion.log_vars[0, i])
    #     params += [criterion.log_vars[0, i]]
    #     criterion.log_vars[0, i].requires_grad = True

    optimizer = PCGrad(AdamW(params, lr=0.001))
    scheduler = ExponentialLR(optimizer._optim, gamma=0.95)
    logs = {"train clf": [], "train recon": [],
            "val clf": [], "val recon": []}
    losses = []
    loop = trange(epochs)
    for epoch in loop:
        train_losses, train_acc = train_pc(model, train_loader, optimizer, criterion)
        val_losses, val_acc = validation(model, val_loader, criterion)
        logs["train clf"].append(train_losses[0])
        logs["train recon"].append(train_losses[1])
        logs["val clf"].append(val_losses[0])
        logs["val recon"].append(val_losses[1])
        loop.set_postfix({"train clf": train_losses[0], "train recon": train_losses[1],
                          "val clf": val_losses[0], "val recon": val_losses[1],
                          "train acc": train_acc, "val acc": val_acc})


        scheduler.step()

    torch.save(model, "model_pc.pt")
    logs = pd.DataFrame(logs)

    logs.plot()

    plt.legend()

    plt.savefig("./logs_pc.jpg")


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
