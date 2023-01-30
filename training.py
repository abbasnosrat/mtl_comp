import numpy as np
import torch
from tqdm.auto import tqdm
import torch.nn as nn


def train(model, data_loader, optimizer, criterion):
    loop = tqdm(data_loader, leave=False)
    r_losses = np.zeros([2])
    r_loss = 0
    r_acc = 0
    model.train()
    loop.set_description("training step")
    for X, y in loop:
        X = X.cuda()
        y = y.cuda()
        clf_out, recon_out = model(X)
        loss, losses, ws = criterion(clf_out, recon_out, y, X)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        r_losses += losses
        r_loss += loss.item()
        with torch.inference_mode():
            r_acc += ((torch.argmax(clf_out, dim=1).flatten() == y.flatten()).sum() / X.shape[0]).item()
        loop.set_postfix({"CCE": losses[0], "MSE": losses[1]})
    r_losses = r_losses / len(data_loader)
    return r_losses, r_loss / len(data_loader), ws, r_acc / len(data_loader)


def train_pc(model, data_loader, optimizer, criterion):
    loop = tqdm(data_loader, leave=False)
    r_losses = np.zeros([2])
    r_loss = 0
    r_acc = 0
    model.train()
    loop.set_description("training step")
    for X, y in loop:
        X = X.cuda()
        y = y.cuda()
        clf_out, recon_out = model(X)
        losses = criterion.compute_loss(clf_out, recon_out, y, X)
        optimizer.zero_grad()
        optimizer.pc_backward(losses)
        optimizer.step()
        r_losses += losses.detach().cpu().numpy()
        losses = losses.detach().cpu().numpy()
        with torch.inference_mode():
            r_acc += ((torch.argmax(clf_out, dim=1).flatten() == y.flatten()).sum() / X.shape[0]).item()
        loop.set_postfix({"CCE": losses[0], "MSE": losses[1]})
    r_losses = r_losses / len(data_loader)
    return r_losses, r_acc / len(data_loader)

def train_uwpc(model, data_loader, optimizer, criterion):
    loop = tqdm(data_loader, leave=False)
    r_losses = np.zeros([2])
    r_loss = 0
    r_acc = 0
    model.train()
    loop.set_description("training step")
    for X, y in loop:
        X = X.cuda()
        y = y.cuda()
        clf_out, recon_out = model(X)
        loss, losses, ws = criterion(clf_out, recon_out, y, X)
        optimizer.zero_grad()
        optimizer.pc_backward(loss)
        optimizer.step()
        r_losses += losses
        r_loss += torch.sum(loss).item()
        with torch.inference_mode():
            r_acc += ((torch.argmax(clf_out, dim=1).flatten() == y.flatten()).sum() / X.shape[0]).item()
        loop.set_postfix({"CCE": losses[0], "MSE": losses[1]})
    r_losses = r_losses / len(data_loader)
    return r_losses, r_loss / len(data_loader), ws, r_acc / len(data_loader)

def validation(model, data_loader, criterion):
    loop = tqdm(data_loader, leave=False)
    r_losses = np.zeros([2])
    r_loss = 0
    r_acc = 0
    model.train()
    loop.set_description("validation step")
    for X, y in loop:
        with torch.inference_mode():
            X = X.cuda()
            y = y.cuda()
            clf_out, recon_out = model(X)
            losses = criterion.compute_loss(clf_out, recon_out, y, X).cpu().numpy()

            r_losses += losses

            r_acc += ((torch.argmax(clf_out, dim=1).flatten() == y.flatten()).sum() / X.shape[0]).item()
        loop.set_postfix({"CCE": losses[0], "MSE": losses[1]})
    r_losses = r_losses / len(data_loader)
    return r_losses, r_acc / len(data_loader)


def train_stl(model, optimizer, data_loader):
    criterion = nn.CrossEntropyLoss()
    r_loss = 0
    r_acc = 0
    for X, y in tqdm(data_loader, total=len(data_loader), leave=False):
        X = X.cuda()
        y = y.cuda()
        clf_out = model(X)
        loss = criterion(clf_out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        r_loss += loss.item()
        with torch.inference_mode():
            r_acc += ((torch.argmax(clf_out, dim=1).flatten() == y.flatten()).sum() / X.shape[0]).item()

    return r_loss / len(data_loader), r_acc / len(data_loader)


def validation_stl(model, data_loader):
    criterion = nn.CrossEntropyLoss()
    r_loss = 0
    r_acc = 0
    for X, y in tqdm(data_loader, total=len(data_loader), leave=False):
        with torch.inference_mode():
            X = X.cuda()
            y = y.cuda()
            clf_out = model(X)
            loss = criterion(clf_out, y)
            r_loss += loss.item()
            r_acc += ((torch.argmax(clf_out, dim=1).flatten() == y.flatten()).sum() / X.shape[0]).item()
    return r_loss / len(data_loader), r_acc / len(data_loader)
