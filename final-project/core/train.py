import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.data.dataset import Dataset

class Trainer():
    #Some baseline training parameters
    ignore_label = 255
    voxel_size = 0.05
    batch_size = 2
    learn_rate = 0.01 #2.4e-1
    weight_decay = 1.0e-4
    momentum = 0.9
    num_workers = 4
    nesterov = False

    def __init__(self, num_classes=17):
        self.num_classes = num_classes

    def train(self, model, dataset, dataroot, device, split='mini', num_epochs=1, 
                last_only=True, rnn=False):

        for param in model.parameters():
            param.requires_grad = True
            param = param.to(device)

        #If last_only is true then only train last layer
        if last_only:
            for name, param in model.named_parameters():
                if name not in ['classifier.linear.weight', 'classifier.linear.bias']:
                    param.requires_grad = False


        #Make optimizer for backpropogation
        optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=self.learn_rate,
                    # momentum=self.momentum,
                    # weight_decay=self.weight_decay,
                    nesterov=self.nesterov)

        #Scheduler for learning
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #             optimizer, lr_lambda=lambda epoch: 1)

        #Loss function
        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_label)

        #Create our training dataset
        train_dataset = Dataset(dataset=dataset, dataroot=dataroot,
                            voxel_size=self.voxel_size, split=split, task='train')

        train_dataloader = DataLoader(
                            train_dataset,
                            batch_size=self.batch_size,
                            collate_fn=train_dataset.collate_fn,
                            num_workers=self.num_workers,
                            pin_memory=True)

        accum_loss, accum_iter, tot_iter = 0, 0, 0
        losses = []
        for epoch in range(num_epochs):

            model.train()
            for data in train_dataloader:
                for key, value in data.items():
                    if key != 'targets':
                        data[key] = value.to(device)

                inputs = data['lidar']
                targets = data['targets'].F.long().to(device, non_blocking=True)

                outputs = model(inputs)

                # if not rnn:
                #     optimizer.zero_grad()

                optimizer.zero_grad()

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                # scheduler.step()

                accum_loss += loss.item()
                accum_iter += 1
                tot_iter += 1

                if tot_iter % 10 == 0 or tot_iter == 1:
                    print('Epoch: {0:d}, Iteration: {1:d}, \
                            Loss: {2:.6f}'.format(epoch, tot_iter, accum_loss/accum_iter))
                    accum_loss, accum_iter = 0, 0

            losses.append(loss)

        return losses





            