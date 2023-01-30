import torch
import torch.nn as nn


class MTLUW(nn.Module):
    def __init__(self):
        super(MTLUW, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(2))

        self.CCE = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.is_regression = torch.tensor([0, 0])

    def compute_loss(self, clf_outputs, recon_outputs, clf_targets, recon_targets):
        losses = torch.zeros(2).to(clf_outputs.device)
        losses[0] = self.CCE(clf_outputs, clf_targets)
        losses[1] = self.MSE(recon_outputs, recon_targets)
        return losses

    def forward(self, clf_outputs, recon_outputs, clf_targets, recon_targets):
        stds = torch.exp(self.log_vars).to(clf_outputs.device)
        self.is_regression = self.is_regression.to(clf_outputs.device)
        ws = 1 / ((self.is_regression + 1) * stds)

        losses = self.compute_loss(clf_outputs, recon_outputs, clf_targets, recon_targets)

        return torch.sum(ws * losses + torch.log(stds)), (losses).detach().cpu().numpy(), ws


class MTLUWR(nn.Module):
    def __init__(self):
        super(MTLUWR, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(2))

        self.CCE = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.is_regression = torch.tensor([0, 0])

    def compute_loss(self, clf_outputs, recon_outputs, clf_targets, recon_targets):
        losses = torch.zeros(2).to(clf_outputs.device)
        losses[0] = self.CCE(clf_outputs, clf_targets)
        losses[1] = self.MSE(recon_outputs, recon_targets)
        return losses

    def forward(self, clf_outputs, recon_outputs, clf_targets, recon_targets):
        stds = torch.exp(self.log_vars).to(clf_outputs.device)
        self.is_regression = self.is_regression.to(clf_outputs.device)
        ws = 2 / ((self.is_regression + 1) * stds)

        losses = self.compute_loss(clf_outputs, recon_outputs, clf_targets, recon_targets)
        # ws = nn.functional.softmax(ws.reshape([1, -1])).flatten()
        return torch.sum(ws * losses + torch.log(1 + stds)), losses.detach().cpu().numpy(), ws


class MTLDWA(nn.Module):
    def __init__(self):
        super(MTLDWA, self).__init__()
        self.CCE = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.loss_t1 = torch.ones(2)
        self.loss_t2 = torch.ones(2)

    def compute_loss(self, clf_outputs, recon_outputs, clf_targets, recon_targets):
        losses = torch.zeros(2).to(clf_outputs.device)
        losses[0] = self.CCE(clf_outputs, clf_targets)
        losses[1] = self.MSE(recon_outputs, recon_targets)
        return losses

    def forward(self, clf_outputs, recon_outputs, clf_targets, recon_targets):
        losses = self.compute_loss(clf_outputs, recon_outputs, clf_targets, recon_targets)
        ws = torch.unsqueeze(self.loss_t1.to(clf_outputs.device) / self.loss_t2.to(clf_outputs.device))
        ws = nn.functional.softmax(ws).flatten()
        self.loss_t2 = self.loss_t1
        self.loss_t1 = losses.detach()
        return torch.sum(ws.to(clf_outputs.device) * losses), (losses).detach().cpu().numpy(), ws


class MTLUWRPC(nn.Module):
    def __init__(self):
        super(MTLUWRPC, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(2))

        self.CCE = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.is_regression = torch.tensor([1, 1])

    def compute_loss(self, clf_outputs, recon_outputs, clf_targets, recon_targets):
        losses = torch.zeros(2).to(clf_outputs.device)
        losses[0] = self.CCE(clf_outputs, clf_targets)
        losses[1] = self.MSE(recon_outputs, recon_targets)
        return losses

    def forward(self, clf_outputs, recon_outputs, clf_targets, recon_targets):
        stds = torch.exp(self.log_vars).to(clf_outputs.device)
        self.is_regression = self.is_regression.to(clf_outputs.device)
        ws = 1 / ((self.is_regression + 1) * stds)

        losses = self.compute_loss(clf_outputs, recon_outputs, clf_targets, recon_targets)
        # ws = nn.functional.softmax(ws.reshape([1, -1])).flatten()
        return ws * losses + torch.log(1 + stds), losses.detach().cpu().numpy(), ws
