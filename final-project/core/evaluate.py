import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.data.dataset import Dataset

class Evaluator():
    #Some baseline training parameters
    ignore_label = 255
    voxel_size = 0.05
    batch_size = 1
    num_workers = 4

    def __init__(self, num_classes=17):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.total_seen = np.zeros(self.num_classes)
        self.total_correct = np.zeros(self.num_classes)
        self.total_positive = np.zeros(self.num_classes)

    def collectIOU(self, output_dict):
        outputs = output_dict['outputs']
        targets = output_dict['targets']
        outputs = outputs[targets != self.ignore_label]
        targets = targets[targets != self.ignore_label]
        
        for i in range(self.num_classes):
            self.total_seen[i] += torch.sum(targets == i).item()
            self.total_correct[i] += torch.sum(
                (targets == i) & (outputs == targets)).item()
            self.total_positive[i] += torch.sum(
                outputs == i).item()

    def mIOU(self):
        ious = []
        
        for i in range(self.num_classes):
            if self.total_seen[i] == 0:
                ious.append(1)
            else:
                cur_iou = self.total_correct[i] / (self.total_seen[i] +
                    self.total_positive[i] - self.total_correct[i])
                ious.append(cur_iou)
        
        return np.mean(ious)

    def evaluate(self, model, dataset, dataroot, device, split='mini', map_labels=False):

        #Reset for mIOU calculations
        self.reset()

        #Loss function
        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_label)

        #Create our validation dataset
        val_dataset = Dataset(dataset=dataset, dataroot=dataroot, 
                        voxel_size=self.voxel_size, split=split, task='val')

        val_dataloader = DataLoader(
                            val_dataset,
                            batch_size=self.batch_size,
                            collate_fn=val_dataset.collate_fn,
                            num_workers=self.num_workers,
                            pin_memory=True)

        losses = []
        with torch.no_grad():
            for data in val_dataloader:
                model.eval()
                
                for key, value in data.items():
                    if key != 'targets':
                        data[key] = value.to(device)

                inputs = data['lidar']
                targets = data['targets'].F.long().to(device, non_blocking=True)

                outputs = model(inputs)

                loss = criterion(outputs, targets)
                losses.append(loss.item())

                #Calculate Mean IOU statistic
                invs = data['inverse_map']
                all_labels = data['targets_mapped']
                _outputs = []
                _targets = []
                for idx in range(invs.C[:, -1].max()+1):
                    cur_scene_pts = (inputs.C[:, -1] == idx).cpu().numpy()
                    cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                    cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()

                    outputs_mapped = outputs[cur_scene_pts][cur_inv].argmax(1)
                    targets_mapped = all_labels.F[cur_label]

                    _outputs.append(outputs_mapped)
                    _targets.append(targets_mapped)
                outputs = torch.cat(_outputs, 0)
                targets = torch.cat(_targets, 0)
                output_dict = {
                    'outputs': outputs,
                    'targets': targets
                }

                self.collectIOU(output_dict)

        return self.mIOU()*100, losses