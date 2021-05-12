from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms
import time
from collections import OrderedDict
import numpy as np
import csv 

print("Preparing data...")

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv_net = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(4, 4))),
            ('relu5', nn.ReLU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(84, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))
    
    def forward(self, x):
        out = self.conv_net(x)
        # print(out.shape)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)

        return out


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        # defines the loss function
        loss = nn.CrossEntropyLoss()
        loss_ = loss(output, target)
        # calculates the backward propagation gradients
        loss_.backward()
        # updates the weights
        optimizer.step()

        # prints accuracy and loss at each epoch
        if batch_idx % 10 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss_.item() 
            ))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)
    ))

    return test_loss, 100. * correct / len(test_loader.dataset)


def run_sgdm():
    # set device
    cuda = True
    use_cuda = cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # load datasets
    batch_size = 128
    test_batch_size = 100
    train_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.MNIST('./data/train', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.MNIST('./data/test', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True
    )

    # set optimizers
    learning_rates = np.arange(0.01, 0.06, 0.01)
    epochs = 5
    momentum = np.arange(0.1, 1, 0.1)
    test_loss, accuracy, tm = [], [], []
    for lr in learning_rates:
        print("==================")
        print("Learning Rate = ", lr)
        print("==================")
        model = LeNet().to(device)
        optimizer =  optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        t0 = time.time()

        for epoch in range(1, epochs+1):
            train(model, device, train_loader, optimizer, epoch)
            loss, acc = test(model, device, test_loader)

        t1 = time.time()
        t = t1-t0
        accuracy.append(acc)
        test_loss.append(round(loss, 2))
        tm.append(round(t, 2))
        print("\nTime taken for training and testing the model is: %s seconds\n" %(t1-t0))

    return learning_rates, accuracy, test_loss, tm

if __name__ == "__main__":
    run_sgdm()
