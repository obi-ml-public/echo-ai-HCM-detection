import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
  
class InceptionModule(nn.Module):

    def __init__(self, in_filters, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5,
                 filters_pool_proj):
        super(InceptionModule, self).__init__()
        self.conv_1x1 = nn.Conv3d(in_filters, filters_1x1, (1, 1, 1))
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm3d(filters_1x1)

        self.conv_3x3 = nn.Conv3d(in_filters, filters_3x3_reduce, (1, 1, 1))
        self.act2 = nn.ReLU()
        self.bn2 = nn.BatchNorm3d(filters_3x3_reduce)
        self.conv_3x3_1 = nn.Conv3d(filters_3x3_reduce, filters_3x3, (3, 3, 3), padding=(1, 1, 1))
        self.act2_1 = nn.ReLU()
        self.bn2_1 = nn.BatchNorm3d(filters_3x3)

        self.conv_5x5 = nn.Conv3d(in_filters, filters_5x5_reduce, (1, 1, 1))
        self.act3 = nn.ReLU()
        self.bn3 = nn.BatchNorm3d(filters_5x5_reduce)
        self.conv_5x5_1 = nn.Conv3d(filters_5x5_reduce, filters_5x5, (3, 3, 3), padding=(1, 1, 1))
        self.act3_1 = nn.ReLU()
        self.bn3_1 = nn.BatchNorm3d(filters_5x5)
        self.conv_5x5_2 = nn.Conv3d(filters_5x5, filters_5x5, (3, 3, 3), padding=(1, 1, 1))
        self.act3_2 = nn.ReLU()

        self.poolproject = nn.MaxPool3d((3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.poolproject_conv3d = nn.Conv3d(in_filters, filters_pool_proj, (1, 1, 1))
        self.act_pool = nn.ReLU()

        self.bn = nn.BatchNorm3d(filters_1x1 + filters_3x3 + filters_5x5 + filters_pool_proj)
        
        self.res = nn.Conv3d(in_filters, filters_1x1 + filters_3x3 + filters_5x5 + filters_pool_proj, (1, 1, 1))
        self.res_bn = nn.BatchNorm3d(filters_1x1 + filters_3x3 + filters_5x5 + filters_pool_proj)
        
    def forward(self, x):
        conv_1x1 = self.bn1(self.act1(self.conv_1x1(x)))

        conv_3x3 = self.bn2(self.act2(self.conv_3x3(x)))
        conv_3x3 = self.bn2_1(self.act2_1(self.conv_3x3_1(conv_3x3)))

        conv_5x5 = self.bn3(self.act3(self.conv_5x5(x)))
        conv_5x5 = self.bn3_1(self.act3_1(self.conv_5x5_1(conv_5x5)))
        conv_5x5 = self.act3_2(self.conv_5x5_2(conv_5x5))

        pool_project = self.act_pool(self.poolproject_conv3d(self.poolproject(x)))

        out = torch.cat((conv_1x1, conv_3x3, conv_5x5, pool_project), dim=1)
        out = self.bn(out)
        
        out = out + self.res_bn(self.res(x))
        return out
    
class Model(nn.Module):

    def __init__(self, filters):
        super(Model, self).__init__()
        self.conv3d_0 = nn.Conv3d(1, filters, (7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3))  # DOUBT
        self.act_0 = nn.ReLU()
        self.bn_0 = nn.BatchNorm3d(filters)
        self.mp_0 = nn.MaxPool3d((3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.conv3d_1 = nn.Conv3d(filters, filters, (1, 1, 1))
        self.act_1 = nn.ReLU()
        self.bn_1 = nn.BatchNorm3d(filters)

        self.conv3d_2 = nn.Conv3d(filters, filters * 3, (3, 3, 3), padding='same')
        self.act_2 = nn.ReLU()
        self.bn_2 = nn.BatchNorm3d(filters * 3)

        self.inception_0 = InceptionModule(in_filters=filters * 3, filters_1x1=filters,
                                           filters_3x3_reduce=int(filters * 1.5),
                                           filters_3x3=filters * 4,
                                           filters_5x5_reduce=int(filters / 4), filters_5x5=int(filters / 2),
                                           filters_pool_proj=int(filters / 2))
        
        in_filters = filters + filters * 4 + int(filters / 2) + int(filters / 2)
        self.inception_1 = InceptionModule(in_filters=in_filters,
                                           filters_1x1=filters * 2, filters_3x3_reduce=filters * 2,
                                           filters_3x3=filters * 3,
                                           filters_5x5_reduce=int(filters / 2), filters_5x5=filters * 3,
                                           filters_pool_proj=filters)

        self.mp_1 = nn.MaxPool3d((1, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1))  # DOUBT

        in_filters = filters * 2 + filters * 3 + filters * 3 + filters
        self.inception_2 = InceptionModule(in_filters=in_filters, filters_1x1=filters * 3,
                                           filters_3x3_reduce=int(filters * 1.5),
                                           filters_3x3=int(filters * 3.25), filters_5x5_reduce=int(filters / 4),
                                           filters_5x5=int(filters * 0.75), filters_pool_proj=filters)

        in_filters = filters * 3 + int(filters * 3.25) + int(filters * 0.75) + filters
        self.inception_3 = InceptionModule(in_filters=in_filters, filters_1x1=int(filters * 2.5),
                                           filters_3x3_reduce=int(filters * 1.75),
                                           filters_3x3=int(filters * 3.5), filters_5x5_reduce=int(filters * 0.375),
                                           filters_5x5=filters,
                                           filters_pool_proj=filters)

        in_filters = int(filters * 2.5) + int(filters * 3.5) + filters + filters
        self.inception_4 = InceptionModule(in_filters=in_filters, filters_1x1=filters * 2,
                                           filters_3x3_reduce=filters * 2,
                                           filters_3x3=filters * 4,
                                           filters_5x5_reduce=int(filters * 0.375), filters_5x5=filters,
                                           filters_pool_proj=filters)

        in_filters = filters * 2 + filters * 4 + filters + filters
        self.inception_5 = InceptionModule(in_filters=in_filters, filters_1x1=int(filters * 1.75),
                                           filters_3x3_reduce=int(filters * 2.25),
                                           filters_3x3=int(filters * 4.5), filters_5x5_reduce=int(filters / 2),
                                           filters_5x5=filters,
                                           filters_pool_proj=filters)

        in_filters = int(filters * 1.75) + int(filters * 4.5) + filters + filters
        self.inception_6 = InceptionModule(in_filters=in_filters, filters_1x1=filters * 4,
                                           filters_3x3_reduce=int(filters * 2.5),
                                           filters_3x3=filters * 5,
                                           filters_5x5_reduce=int(filters / 2), filters_5x5=filters * 2,
                                           filters_pool_proj=filters * 2)

        self.mp_2 = nn.MaxPool3d((1, 3, 3), stride=(2, 2, 2))

        in_filters = filters * 4 + filters * 5 + filters * 2 + filters * 2
        self.inception_7 = InceptionModule(in_filters=in_filters, filters_1x1=filters * 4,
                                           filters_3x3_reduce=int(filters * 2.5),
                                           filters_3x3=filters * 5,
                                           filters_5x5_reduce=int(filters / 2), filters_5x5=filters * 2,
                                           filters_pool_proj=filters * 2)

        in_filters = filters * 4 + filters * 5 + filters * 2 + filters * 2
        self.inception_8 = InceptionModule(in_filters=in_filters, filters_1x1=filters * 6,
                                           filters_3x3_reduce=filters * 3,
                                           filters_3x3=filters * 6,
                                           filters_5x5_reduce=int(filters * 0.75), filters_5x5=filters * 2,
                                           filters_pool_proj=filters * 2)
    
        self.mp_3 = nn.MaxPool3d((1, 2, 2))

        in_filters = filters * 6 + filters * 6 + filters * 2 + filters * 2
        self.inception_9 = InceptionModule(in_filters=in_filters, filters_1x1=filters * 6,
                                           filters_3x3_reduce=filters * 3,
                                           filters_3x3=filters * 6,
                                           filters_5x5_reduce=int(filters * 0.75), filters_5x5=filters * 2,
                                           filters_pool_proj=filters * 2)
        
        self.inception_10 = InceptionModule(in_filters=in_filters, filters_1x1=filters * 6,
                                           filters_3x3_reduce=filters * 3,
                                           filters_3x3=filters * 6,
                                           filters_5x5_reduce=int(filters * 0.75), filters_5x5=filters * 2,
                                           filters_pool_proj=filters * 2)
        
        
        self.inception_11 = InceptionModule(in_filters=in_filters, filters_1x1=filters * 8,
                                           filters_3x3_reduce=filters * 4,
                                           filters_3x3=filters * 8,
                                           filters_5x5_reduce=int(filters * 1), filters_5x5=filters * 3,
                                           filters_pool_proj=filters * 3)
        
        

        self.dp = nn.Dropout(0.1)
        
        self.ln_x = nn.Linear(filters * 8 + filters * 8 + filters * 3 + filters * 3, 100)
        self.bn_x = nn.BatchNorm1d(100)
        
        self.ln_a = nn.Linear(1, 100)
        self.bn_a = nn.BatchNorm1d(100)
        
        self.ln_b = nn.Linear(1, 100)
        self.bn_b = nn.BatchNorm1d(100)
        
        self.bn = nn.BatchNorm1d(100 * 3)
        self.linear = nn.Linear(100 * 3, 1)
        self.act = nn.Sigmoid()

    def forward(self, x, inputScale, inputInvScale):
        x = self.act_0(self.conv3d_0(x))
        x = self.mp_0(self.bn_0(x))

        x = self.bn_1(self.act_1(self.conv3d_1(x)))
        x = self.bn_2(self.act_2(self.conv3d_2(x)))

        x = self.inception_0(x)
        x = self.inception_1(x)
        x = self.mp_1(x)

        x = self.inception_2(x)
        x = self.inception_3(x)
        x = self.inception_4(x)
        x = self.inception_5(x)
        x = self.inception_6(x)
        x = self.mp_2(x)

        x = self.inception_7(x)
        x = self.inception_8(x)
        x = self.mp_3(x)

        x = self.inception_9(x)
        x = self.inception_10(x)
        x = self.inception_11(x)

        x = torch.mean(x, [2, 3, 4])
        
        x = self.dp(x)
        x = self.ln_x(x)
        x = self.bn_x(x)
        
        a = inputScale
        a = self.ln_a(a)
        a = self.bn_a(a)
        a = x * a
        
        b = inputInvScale
        b = self.ln_b(b)
        b = self.bn_b(b)
        b = x * b
        
        x = torch.cat([x, a, b], dim=1)
        x = self.bn(x)
        
        
        xa = self.linear(x)
        out = self.act(xa)

        return out
