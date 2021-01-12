import os
import numpy as np
from core.data.mappings import *

split_map = {
    'trainval' : {'train': ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10'], 'val': ['08']},
    'mini' : {'train' : ['00', '01'], 'val': ['08']},
    'test' : {'test': ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']}
}

'''
Collects all the proper files for data and labels.
Returns a dictionary of data->data files, labels->label files,
and the size of the dataset.
'''
def getFiles(dataroot, split='mini', task='train'):
    #Make sure split is valid
    assert split in split_map.keys()
    assert task in ['train', 'val']

    size = 0 #Number of keyframes

    pc_data_files = [] #Contains lidar data file paths
    pc_label_files = [] #Contains lidar label file paths

    #Extract lidar data and labels
    #We are making the assumption that there are the
    #same number of label and data files... which there is
    for scene_name in split_map[split][task]:
        root = dataroot + '/data/' + scene_name
        
        data_root = root+'/velodyne/'
        label_root = root+'/labels/'
        for filename in os.listdir(data_root):
            pc_data_files.append(data_root+filename)
            pc_label_files.append(label_root+filename.replace('.bin', '.label'))

            size += 1

    return {'data':pc_data_files, 'labels':pc_label_files}, size