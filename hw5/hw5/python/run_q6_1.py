#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 13:15:22 2020

@author: akshit
"""
# Define the netowrk
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# train_data = scipy.io.loadmat('../data/nist36_train.mat')
# test_data = scipy.io.loadmat('../data/nist36_test.mat')

# train_x, train_y = train_data['train_data'], train_data['train_labels']
# test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 100
# # pick a batch size, learning rate
# batch_size = 32
# learning_rate = 1e-2
# hidden_size = 64

# device = torch.device('cpu')

# trainset_x = torch.from_numpy(train_x).float()
# trainset_y = torch.from_numpy(train_y).type(torch.LongTensor)
# trainloader = DataLoader(TensorDataset(trainset_x,trainset_y),batch_size=batch_size,shuffle=True)

# testset_x = torch.from_numpy(test_x).type(torch.float32)
# testset_y = torch.from_numpy(test_y).type(torch.LongTensor)
# testloader = DataLoader(TensorDataset(testset_x,testset_y),batch_size=batch_size,shuffle=False)

# class Net(nn.Module):
#     def __init__(self, input_dim, hid_size, output_dim):
#         super(Net,self).__init__()
#         # an affine operation: y = Wx + b
#         self.fc1 = nn.Linear(input_dim,hid_size) # Applies a linear transformation to the incoming data
#         self.fc2 = nn.Linear(hid_size,output_dim)

#     def forward(self, x):
#         # Max pooling over a (2,2) window
#         act_x = F.sigmoid(self.fc1(x))
#         pred_val = self.fc2(act_x)
#         return pred_val

# net = Net(train_x.shape[1], hidden_size,train_y.shape[1])
# opt = optim.SGD(net.parameters(), lr = learning_rate, momentum = 0.9) # Stochastic Gradient Descent

# train_loss = []
# train_acc = []

# for itr in range(max_iters):
#     total_loss = 0
#     avg_acc = 0
#     for im_data in trainloader:
#         inputs = torch.autograd.Variable(im_data[0])
#         labels = torch.autograd.Variable(im_data[1])
#         act = torch.max(labels,1)[1]

#         pred = net(inputs)
#         loss = nn.functional.cross_entropy(pred,act)
#         total_loss += loss.item()
#         pred_val = torch.max(pred,1)[1]

#         avg_acc += pred_val.eq(act.data).cpu().sum().item()

#         # backward
#         loss.backward()
#         opt.step()
#         opt.zero_grad()

#     avg_acc = avg_acc / train_y.shape[0]
#     train_loss.append(total_loss)
#     train_acc.append(avg_acc)

#     if itr % 2 == 0:
#         print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,avg_acc))

# print('Training Accuracy: {}'.format(train_acc[-1]))
# torch.save(net.state_dict(),"q6_1_nn.pkl")
# plt.figure("Accuracy")
# plt.plot(range(max_iters), train_acc)
# plt.xlabel('Epoch')
# plt.ylabel('Training Accuracy')
# plt.show()

# plt.figure("Cross-Entropy Loss")
# plt.plot(range(max_iters), train_loss)
# plt.xlabel('Epoch')
# plt.ylabel('Training Cross-Entropy Loss')
# plt.show()

# test_val = 0
# for test_data in testloader:
#     # get the inputs
#     inputs = torch.autograd.Variable(test_data[0])
#     labels = torch.autograd.Variable(test_data[1])
#     act = torch.max(labels, 1)[1]

#     # get output
#     pred = net(inputs)
#     loss = nn.functional.cross_entropy(pred, act)

#     pred_val = torch.max(pred, 1)[1]
#     test_val += pred_val.eq(act.data).cpu().sum().item()

# test_acc = test_val/test_y.shape[0]

# print('Test accuracy: {}'.format(test_acc))

# 6.1.2
batch_size = 32
learning_rate = 0.01
epochs = 4
tf = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.15,), (0.3,))])

trainloader = DataLoader(torchvision.datasets.MNIST('../data/',train=True, download= True,
                        transform = tf) ,batch_size = batch_size, shuffle = True)

testloader = DataLoader(torchvision.datasets.MNIST('../data/',train=False, download = True,
                        transform = tf) ,shuffle = False)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5,stride=1)
        self.conv2 = nn.Conv2d(10,20,kernel_size=5,stride=1)
        self.fc1 = nn.Linear(4*4*20,100) # Applies a linear transformation to the incoming data
        self.fc2 = nn.Linear(100,10)

    def forward(self, x):
        act_x = F.relu(self.conv1(x))
        act_x = F.max_pool2d(act_x,2,2)
        act_x = F.relu(self.conv2(act_x))
        act_x = F.max_pool2d(act_x,2,2)
        act_x = act_x.view(-1,4*4*20)
        act_x = F.relu(self.fc1(act_x))
        act_x  = self.fc2(act_x)

        return act_x

net = ConvNet()
opt = optim.SGD(net.parameters(), lr = learning_rate, momentum = 0.9) # Stochastic Gradient Descent

train_loss = []
train_acc = []

for itr in range(max_iters):
    total_loss = 0
    avg_acc = 0

    for im_data in trainloader:
        inputs = torch.autograd.Variable(im_data[0])
        labels = torch.autograd.Variable(im_data[1])

        pred = net(inputs)
        act = torch.max(pred,1)[1]
        loss = nn.functional.cross_entropy(pred,act)
        total_loss += loss.item()
        pred_val = torch.max(pred,1)[1]

        avg_acc += pred_val.eq(act.data).cpu().sum().item()

        # backward
        loss.backward()
        opt.step()
        opt.zero_grad()

    avg_acc = avg_acc / len(trainloader)
    train_loss.append(total_loss)
    train_acc.append(avg_acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,avg_acc))
