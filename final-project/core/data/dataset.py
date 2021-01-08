import torch
import numpy as np

from torchsparse import SparseTensor
from torchsparse.utils import sparse_collate_fn, sparse_quantize

from core.data.mappings import *
import core.data.nuscenes
import core.data.kitti

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, dataroot, voxel_size, split='mini', task='train', 
                    num_points=None, augment=False, mean=False):

        #Copy input parameters
        self.task = task
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.augment = augment
        self.mean = mean

        if dataset == 'nuscenes':
            self.files, self.size = nuscenes.getFiles(dataroot+'/'+dataset, split, task)
            self.labelmap = lambda x : nuscenes2model[x]

        elif dataset == 'kitti':
            self.files, self.size = kitti.getFiles(dataroot+'/'+dataset, split, task)
            self.label_map = lambda x : kitti2model[x]

    def augment(self, pt_cloud):
        theta = np.random.uniform(0, 2*np.pi)
        scale = np.random.uniform(0.95, 1.05)

        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])

        return np.dot(pt_cloud[:,:3], R)*scale_factor

    def __len__(self):
        #The size of the dataset is the number of keyframes
        return self.size
    
    def __getitem__(self, idx):
        #Get the point cloud data
        #Note: The fourth dimension is intensity
        pt_cloud = np.fromfile(self.files['data'][idx], 
                                dtype=np.float32).reshape((-1,5))[:,:4]

        #Augment that data
        if self.augment and train in self.split:
            pt_cloud = self.augment(pt_cloud)

        #Get the segmentation annotations
        pc_labels = np.fromfile(self.files['labels'][idx], dtype=np.uint8)

        #Map the labels
        for i, l in enumerate(pc_labels):
            l = self.labelmap(l)
            pc_labels[i] = l

        #Create the voxelized point cloud coordinates
        voxels = np.round(pt_cloud / self.voxel_size).astype(np.int32)

        #inds: unique coordinates in the voxelized point cloud
        #labels_vox: voxel labels, if there is a conflict -> label=255
        #inverse_map: the ith value is the voxel coordinate of the ith point
        inds, vox_labels, inverse_map = sparse_quantize(voxels, pt_cloud, pc_labels,
                                                    return_index=True, 
                                                    return_invs=True,
                                                    ignore_label=255)
        
        #If we have too many points scrap some of them
        if self.num_points != None:
            if len(inds) > self.num_points:
                inds = np.random.choice(inds, self.num_points, replace=False)

        #Just get the unique voxel coordinates
        coords = torch.from_numpy(voxels[inds])

        #We want the feature of each voxel to be the mean value of the
        #features in that voxel. Unfortunately this is slow to do outside
        #of the quantize function.
        if self.mean:
            features = torch.zeros((len(coords), 4), dtype=torch.float32)
            num_points = torch.zeros((len(coords), 1))
            for i,feat in enumerate(pt_cloud):
                features[inverse_map[i]] += feat
                num_points[inverse_map[i]] += 1

            features = features/num_points
        #Or in this case just pick one of them
        else:
            features = torch.from_numpy(pt_cloud[inds])

        #Labels associated with unique voxels
        labels = torch.from_numpy(pc_labels[inds])

        #Convert point cloud and labels to tensors
        pt_cloud = torch.from_numpy(pt_cloud)
        pc_labels = torch.from_numpy(pc_labels)

        #Create the sparse tensors
        lidar = SparseTensor(features, coords) #features
        targets = SparseTensor(labels, coords) #labels
        targets_mapped = SparseTensor(pc_labels, voxels) #
        inverse_map = SparseTensor(inverse_map, voxels)

        return {
                'lidar': lidar,
                'targets': targets,
                'targets_mapped': targets_mapped,
                'inverse_map': inverse_map,
                }

    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)