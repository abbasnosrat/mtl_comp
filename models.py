import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, in_shape=[1, 28, 28], num_classes=10):
        super(Net, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_shape[0], 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.decoder = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(64, in_shape[0], kernel_size=3, stride=1))
        dim = self.encoder(torch.randn([1, *in_shape])).flatten(start_dim=1).shape[-1]
        self.classifier = nn.Sequential(nn.Flatten(start_dim=1),
                                        nn.Linear(dim, 128), nn.ReLU(),
                                        nn.Linear(128, num_classes))

    def forward(self, X):
        z = self.encoder(X)
        clf_out = self.classifier(z)
        recon_out = self.decoder(z)
        return clf_out, recon_out


class StlNet(nn.Module):
    def __init__(self, in_shape=[1, 28, 28], num_classes=10):
        super(StlNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_shape[0], 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU())

        dim = self.encoder(torch.randn([1, *in_shape])).flatten(start_dim=1).shape[-1]
        self.classifier = nn.Sequential(nn.Flatten(start_dim=1),
                                        nn.Linear(dim, 128), nn.ReLU(),
                                        nn.Linear(128, num_classes))

    def forward(self, X):
        z = self.encoder(X)
        clf_out = self.classifier(z)

        return clf_out
