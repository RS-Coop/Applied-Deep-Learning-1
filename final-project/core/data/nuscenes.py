import torch
import numpy as np

from nuscenes import NuScenes
from nuscenes.utils import splits as nu_splits

from torchsparse import SparseTensor
from torchsparse.utils import sparse_collate_fn, sparse_quantize

from core.data.mappings import *

class NuScenesDataset(torch.utils.data.Dataset):
    split_map = {
        'trainval' : {'train':nu_splits.train, 'val':nu_splits.val},
        'mini' : {'train':nu_splits.mini_train, 'val':nu_splits.mini_val},
        'test' : {'test':nu_splits.test}
    }
    def __init__(self, dataroot, voxel_size, split='mini',
                task='train', num_points=None, augment=False, mean=False):

        #Make sure split is valid
        assert split in self.split_map.keys()
        assert task in ['train', 'test', 'val']

        #Copy input parameters
        self.split = split
        self.task = task
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.augment = augment
        self.mean = mean

        #Open nuscenes split
        nusc = NuScenes(version='v1.0-'+self.split, dataroot=dataroot, verbose=0)
        self.size = 0 #Number of keyframes

        self.pc_data_files = [] #Contains lidar data file paths
        self.pc_label_files = [] #Contains lidar label file paths

        #Extract lidar data and labels so that they match in terms of index
        start = 0
        for scene_name in sorted(self.split_map[self.split][self.task]):
            for i,scene in enumerate(nusc.scene[start:], start=start):
                if scene_name == scene['name']:
                    token = scene['first_sample_token']
                    while token != '':
                        sample = nusc.get('sample', token)

                        sample_data_token = sample['data']['LIDAR_TOP']

                        data_path = nusc.get_sample_data_path(sample_data_token)
                        label_file = nusc.get('lidarseg', sample_data_token)['filename']

                        self.pc_data_files.append(data_path)
                        self.pc_label_files.append(dataroot + '/' + label_file)

                        token = sample['next']
                        self.size += 1

                    start = i
                    break

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
        pt_cloud = np.fromfile(self.pc_data_files[idx], 
                                dtype=np.float32).reshape((-1,5))[:,:4]

        #Augment that data
        if self.augment and train in self.split:
            pt_cloud = self.augment(pt_cloud)

        #Get the segmentation annotations
        pc_labels = np.fromfile(self.pc_label_files[idx], dtype=np.uint8)

        #Map the labels
        for i, l in enumerate(pc_labels):
            pc_labels[i] = nuscenes2model[l]

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
        #features in that voxel.
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