import numpy as np

from nuscenes import NuScenes
from nuscenes.utils import splits as nu_splits

from core.data.mappings import *

split_map = {
    'trainval' : {'train':nu_splits.train, 'val':nu_splits.val},
    'mini' : {'train':nu_splits.mini_train, 'val':nu_splits.mini_val},
    'test' : {'test':nu_splits.test}
}

'''
Collects all the proper files for data and labels.
Returns a dictionary of data->data files, labels->label files,
and the size of the dataset.
'''
def getFiles(self, dataroot, split='mini', task='train'):
    #Make sure split is valid
    assert split in split_map.keys()
    assert task in ['train', 'test', 'val']
    
    #Open nuscenes split
    nusc = NuScenes(version='v1.0-'+split, dataroot=dataroot, verbose=0)
    size = 0 #Number of keyframes

    pc_data_files = [] #Contains lidar data file paths
    pc_label_files = [] #Contains lidar label file paths

    #Extract lidar data and labels so that they match in terms of index
    start = 0
    for scene_name in sorted(split_map[split][task]):
        for i,scene in enumerate(nusc.scene[start:], start=start):
            if scene_name == scene['name']:
                token = scene['first_sample_token']
                while token != '':
                    sample = nusc.get('sample', token)

                    sample_data_token = sample['data']['LIDAR_TOP']

                    data_path = nusc.get_sample_data_path(sample_data_token)
                    label_file = nusc.get('lidarseg', sample_data_token)['filename']

                    pc_data_files.append(data_path)
                    pc_label_files.append(dataroot + '/' + label_file)

                    token = sample['next']
                    size += 1

                start = i
                break

    return {'data':pc_data_files, 'labels':pc_label_files}, size