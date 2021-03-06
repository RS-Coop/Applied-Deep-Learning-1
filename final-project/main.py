import torch
import numpy as np
import json

from core.model import SPVStd, SPVRnn
from core.train import Trainer
from core.evaluate import Evaluator

#Global model choices
net_id = '@20' #This is the smaller SPVNAS model
rnn = False
split = 'mini'
dataroot = 'core/data/datasets'
dataset = 'nuscenes'
num_epochs = 3
last_only = True

if __name__=='__main__':
    #Get compute device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'

    print('Training using device: {}\n'.format(device))

    #Get Network configurations
    net_config = json.load(open('core/pretrained/net{}.config'.format(net_id)))
    params = torch.load('core/pretrained/init{}'.format(net_id),
                        map_location=device)['model']

    #If working with nuscenes change number of classes
    #and delete the classifier weights as they wont match.
    if dataset == 'nuscenes':
        net_config['num_classes'] = 16
        del params['classifier.linear.weight']
        del params['classifier.linear.bias']
        
    #Build the network
    if rnn:
        model = SPVRnn(net_config, params, device)
        model.build('GRU')

        print('RNN SVPNAS network built.\n')
    else:
        model = SPVStd(net_config, params, device)

        print('Standard SPVNAS network built.\n')

    model.to(device)

    #Train the network
    print('Training network on data split {}\n'.format(split))
    T = Trainer()
    T.train(model, dataset, dataroot, device, split=split, 
            num_epochs=num_epochs, last_only=last_only)

    print('Training finished, hopefuly that went well, saving model parameters.\n')
    # torch.save(model.state_dict(), 'core/pretrained/'+net_id+'-split-'+split+'-rnn-'+str(rnn))

    #Evaluate network
    print('Evaluating network.\n')
    E = Evaluator()
    miou, _ = E.evaluate(model, dataset, dataroot, device, split=split)

    print('Validation set mean IOU: {}'.format(miou))

    print('Goodbye!')


    