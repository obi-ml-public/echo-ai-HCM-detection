import torch
import torch.nn as nn


class multi_conv2d_new(nn.Module):

    def __init__(self, in_filters, num_filters):
        super(multi_conv2d_new, self).__init__()
        self.a1 = nn.Conv2d(in_filters, int(num_filters/1.5), kernel_size=(3,3), padding='same')
        self.act_a1 = nn.ReLU()
        self.bn_a1 = nn.BatchNorm2d(int(num_filters/1.5), affine=True)
        self.a2 = nn.Conv2d(int(num_filters/1.5), num_filters, kernel_size=(1,1), padding='same')
        self.act_a2 = nn.ReLU()
        self.bn_a2 = nn.BatchNorm2d(num_filters, affine=True)
        
        self.b1 = nn.Conv2d(in_filters, int(num_filters/1.5), kernel_size=(7,7), padding='same')
        self.act_b1 = nn.ReLU()
        self.bn_b1 = nn.BatchNorm2d(int(num_filters/1.5), affine=True)
        self.b2 = nn.Conv2d(int(num_filters/1.5), num_filters, kernel_size=(3,3), padding='same')
        self.act_b2 = nn.ReLU()
        self.bn_b2 = nn.BatchNorm2d(num_filters, affine=True)
        self.b3 = nn.Conv2d(num_filters, num_filters, kernel_size=(1,1), padding='same')
        self.act_b3 = nn.ReLU()
        self.bn_b3 = nn.BatchNorm2d(num_filters, affine=True)
        
        self.c1 = nn.Conv2d(in_filters, num_filters, kernel_size=(1,1), padding='same')
        self.act_c1 = nn.ReLU()
        self.bn_c1 = nn.BatchNorm2d(num_filters, affine=True)

        self.res  = nn.Conv2d(in_filters, num_filters * 3, kernel_size=1, stride=1)
        
    def forward(self, x):
        a = self.act_a1(self.a1(x))
        a = self.bn_a1(a)
        a = self.act_a2(self.a2(a))
        a = self.bn_a2(a)
        
        b = self.act_b1(self.b1(x))
        b = self.bn_b1(b)
        b = self.act_b2(self.b2(b))
        b = self.bn_b2(b)
        b = self.act_b3(self.b3(b))
        b = self.bn_b3(b)
        
        c = self.act_c1(self.c1(x))
        c = self.bn_c1(c)
        
        out = torch.cat((a,b,c), dim=1)
        
        out = out + self.res(x)
        return out
    
class Convolutional2d_new(nn.Module):

    def __init__(self, initial_filters=64):
        super(Convolutional2d_new, self).__init__()
        
        self.conv1 = nn.Conv2d(1, initial_filters, kernel_size=(7,3), stride=(2,1))
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(initial_filters, affine=True)
        
        self.multi_conv2d_1 = multi_conv2d_new(initial_filters, initial_filters)
        self.multi_conv2d_2 = multi_conv2d_new(int(initial_filters*3), initial_filters)
        self.mp1 = nn.MaxPool2d((3, 1))
        
        self.multi_conv2d_3 = multi_conv2d_new(int(initial_filters*3), int(initial_filters*1.5))
        self.multi_conv2d_4 = multi_conv2d_new(int(initial_filters*1.5*3), int(initial_filters*1.5))
        self.mp2 = nn.MaxPool2d((3, 1))
        
        self.multi_conv2d_5 = multi_conv2d_new(int(initial_filters*1.5*3), int(initial_filters*2))
        self.multi_conv2d_6 = multi_conv2d_new(int(initial_filters*2*3), int(initial_filters*2))
        self.mp3 = nn.MaxPool2d((2, 1))
        
        self.multi_conv2d_7 = multi_conv2d_new(int(initial_filters*2*3), int(initial_filters*3))
        self.multi_conv2d_8 = multi_conv2d_new(int(initial_filters*3*3), int(initial_filters*3))
        self.multi_conv2d_9 = multi_conv2d_new(int(initial_filters*3*3), int(initial_filters*4))
        self.mp4 = nn.MaxPool2d((2, 1))
        
        self.multi_conv2d_10 = multi_conv2d_new(int(initial_filters*4*3), int(initial_filters*5))
        self.multi_conv2d_11 = multi_conv2d_new(int(initial_filters*5*3), int(initial_filters*6))
        self.multi_conv2d_12 = multi_conv2d_new(int(initial_filters*6*3), int(initial_filters*7))
        self.mp5 = nn.MaxPool2d((2, 1))
        
        self.multi_conv2d_13 = multi_conv2d_new(int(initial_filters*7*3), int(initial_filters*8))
        self.multi_conv2d_14 = multi_conv2d_new(int(initial_filters*8*3), int(initial_filters*8))
        self.multi_conv2d_15 = multi_conv2d_new(int(initial_filters*8*3), int(initial_filters*8))
        self.mp6 = nn.MaxPool2d((2, 1))
        
        self.multi_conv2d_16 = multi_conv2d_new(int(initial_filters*8*3), int(initial_filters*12))
        self.multi_conv2d_17 = multi_conv2d_new(int(initial_filters*12*3), int(initial_filters*14))
        self.multi_conv2d_18 = multi_conv2d_new(int(initial_filters*14*3), int(initial_filters*16))
        
        self.multi_conv2d_19 = multi_conv2d_new(int(initial_filters*16*3), int(initial_filters*16))
        self.multi_conv2d_20 = multi_conv2d_new(int(initial_filters*16*3), int(initial_filters*16))

        self.dp = nn.Dropout(0.0)
        self.linear = nn.Linear(int(initial_filters*16*3), 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(x)
        
        x = self.multi_conv2d_1(x)
        x = self.multi_conv2d_2(x)
        x = self.mp1(x)
        
        x = self.multi_conv2d_3(x)
        x = self.multi_conv2d_4(x)
        x = self.mp2(x)
        
        x = self.multi_conv2d_5(x)
        x = self.multi_conv2d_6(x)
        x = self.mp3(x)
        
        x = self.multi_conv2d_7(x)
        x = self.multi_conv2d_8(x)
        x = self.multi_conv2d_9(x)
        x = self.mp4(x)
        
        x = self.multi_conv2d_10(x)
        x = self.multi_conv2d_11(x)
        x = self.multi_conv2d_12(x)
        x = self.mp5(x)
        
        x = self.multi_conv2d_13(x)
        x = self.multi_conv2d_14(x)
        x = self.multi_conv2d_15(x)
        x = self.mp6(x)
        
        x = self.multi_conv2d_16(x)
        x = self.multi_conv2d_17(x)
        x = self.multi_conv2d_18(x)
        
        x = self.multi_conv2d_19(x)
        x = self.multi_conv2d_20(x)

        x = torch.mean(x, [2, 3])
        
        x = self.dp(x)
        x = self.linear(x)
        
        out = self.act(x)
        return out
    
    
