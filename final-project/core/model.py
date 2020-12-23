import torchsparse
from torchsparse.point_tensor import PointTensor

from core.modules.utils import *
from core.modules.spvnas import SPVNAS
# from core.sparse_rnn import sparseGRU, sparseLSTM

class SPVStd(SPVNAS):
    def __init__(self, net_config, params, device):
        super().__init__(net_config['num_classes'],
                        macro_depth_constraint=1,
                        pres=net_config['pres'],
                        vres=net_config['vres'])

        self.to(device)
        self.manual_select(net_config)
        self.determinize()
        self.load_state_dict(params, strict=False)

class SPVRnn(SPVStd):
    def __init__(self, net_config, params, device):
        super().__init__(net_config, params, device)

        self.device = device

    #Add specified sparse rnn layer to model
    def build(self, type, num_layers=1):
        #self.output_channels[-1] is final output channels

        if type == 'GRU':
            self.rnn = sparseGRU()
        elif type == 'LSTM':
            self.rnn = sparseLSTM()
        else:
            raise ValueError('Invalid recurrent layer type. \
                            Must be one of "GRU" or "LSTM".')

        self.to(self.device)

    #Overwrite current forward function to include
    #recurent layer.
    def forward(self, x):
        # x: SparseTensor z: PointTensor
        z = PointTensor(x.F, x.C.float())
        #x0 = initial_voxelize(z, self.pres, self.vres)
        x0 = point_to_voxel(x, z)
        
        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z)
        z0.F = z0.F  #+ self.point_transforms[0](z.F)

        x1 = point_to_voxel(x0, z0)
        x1 = self.downsample[0](x1)
        x2 = self.downsample[1](x1)
        x3 = self.downsample[2](x2)
        x4 = self.downsample[3](x3)

        # point transform 32 to 256
        z1 = voxel_to_point(x4, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)

        y1 = point_to_voxel(x4, z1)
        y1.F = self.dropout(y1.F)
        y1 = self.upsample[0].transition(y1)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.upsample[0].feature(y1)

        #print('y1', y1.C)
        y2 = self.upsample[1].transition(y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.upsample[1].feature(y2)
        # point transform 256 to 128
        z2 = voxel_to_point(y2, z1)
        z2.F = z2.F + self.point_transforms[1](z1.F)

        y3 = point_to_voxel(y2, z2)
        y3.F = self.dropout(y3.F)
        y3 = self.upsample[2].transition(y3)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.upsample[2].feature(y3)

        y4 = self.upsample[3].transition(y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.upsample[3].feature(y4)
        z3 = voxel_to_point(y4, z2)
        z3.F = z3.F + self.point_transforms[2](z2.F)

        '''
        The RNN should go here after we have extracted all the meaningful
        features. The question now is how should we do the RNN? Should we
        just voxelize->rnn->devoxelize. Should we have two branches where
        one is a regular rnn for the points. Should we have a point voxel
        convolution inside the rnn?
        '''
        r1 = point_to_voxel(y4, z3)
        #r1 is a sparse tensor with features of size Nx48
        # r1 = self.rnn(r1) #Pass the sparse tensor through rnn
        final = voxel_to_point(r1, z3)

        self.classifier.set_in_channel(final.F.shape[-1])
        out = self.classifier(final.F)

        return out