from __future__ import print_function, division
import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from torchvision import transforms

import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFile
import copy
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train_model(model, criterion, optimizer, lr_scheduler,dset_loaders,dset_sizes, num_epochs=100,use_gpu =0):
    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['training', 'testing']:
            if phase == 'training':
                mode='train'
                optimizer = lr_scheduler(optimizer, epoch)
                model.train()  # Set model to training mode
            else:
                model.eval()
                mode='val'

            running_loss = 0.0
            running_corrects = 0

            counter=0
            # Iterate over data.
            for data in dset_loaders[phase]:
                inputs, labels = data
                # wrap them in Variable
                if use_gpu:
                    try:
                        inputs, labels = Variable(inputs.float().cuda()),                             
                        Variable(labels.long().cuda())
                    except:
                        # print(inputs,labels)
                        print("exception")
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # Set gradient to zero to delete history of computations in previous epoch. Track operations so that differentiation can be done automatically.
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                
                loss = criterion(outputs, labels)
                # print('loss done')                
                # Just so that you can keep track that something's happening and don't feel like the program isn't running.
                # if counter%10==0:
                #     print("Reached iteration ",counter)
                counter+=1

                # backward + optimize only if in training phase
                if phase == 'training':
                    # print('loss backward')
                    loss.backward()
                    # print('done loss backward')
                    optimizer.step()
                    # print('done optim')
                # print evaluation statistics
                try:
                    # running_loss += loss.data[0]
                    running_loss += loss.item()
                    # print(labels.data)
                    # print(preds)
                    running_corrects += torch.sum(preds == labels.data)
                    # print('running correct =',running_corrects)
                except:
                    print('unexpected error, could not calculate loss or do a sum.')
            print('trying epoch loss')
            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects.item() / float(dset_sizes[phase])
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))


            # deep copy the model
            if phase == 'testing':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(model)
                    print('new best accuracy = ',best_acc)
    print('Training complete')
    print('Best val Acc: {:4f}'.format(best_acc))
    print('returning and looping back')
    return best_model

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=30):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def train_model1(model, criterion,train_loader,test_loader,device, learning_rate,num_epochs=100,param_group=True):

    best_model = model
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9,
                                  weight_decay=0.001)

    for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

      
 
            model.train()  # Set model to training mode


            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in train_loader:
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)

                
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                try:
                    running_loss += loss.item()
                    running_corrects += torch.sum(preds == labels.data)
                except:
                    print('unexpected error, could not calculate loss or do a sum.')
            print('trying epoch loss')
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.item() / float(len(train_loader.dataset))
            print('Loss: {:.4f} Acc: {:.4f}'.format(
                epoch_loss, epoch_acc))

    print('Training complete')
    accuracy = evaluate_accuracy(test_loader, model, device)
    print(f'Validation Accuracy: {accuracy * 100:.2f}%')
    return best_model

def evaluate_accuracy(data_loader, net, device):
    net.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    print("here6")

    with torch.no_grad():  # No need to compute gradients during evaluation
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            print("predicted: ",predicted)
            print("labels: ",labels)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


