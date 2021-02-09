import torch
import numpy as np
import json

from core.model import SPVStd, SPVRnn
from core.evaluate import Evaluator

#Global model choices
net_id = '@20' #This is the smaller SPVNAS model
rnn = False
split = 'trainval'
dataroot = 'core/data/datasets'
dataset = 'kitti'

if __name__=='__main__':
    #Get compute device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'

    print('Using device: {}\n'.format(device))

    #Get Network configurations
    net_config = json.load(open('core/pretrained/net{}.config'.format(net_id)))
    params = torch.load('core/pretrained/init{}'.format(net_id),
                        map_location=device)['model']

    #Build the network
    if rnn:
        model = SPVRnn(net_config, params, device)
        model.build('GRU')

        print('RNN SVPNAS network built.\n')
    else:
        model = SPVStd(net_config, params, device)

        print('Standard SPVNAS network built.\n')

    model.to(device)

    #Evaluate network
    print('Evaluating network.\n')
    E = Evaluator(num_classes=num_classes)
    miou, _ = E.evaluate(model, dataset, dataroot, device, split=split)

    print('Validation set mean IOU: {}'.format(miou))

    print('Goodbye!')