import torch
import torch.nn as nn
import torch.nn.functional as F

class TempoNet(nn.Module):
    def __init__(self):
        super(TempoNet, self).__init__()
        # Short Conv NN x3

        # Spectros come in at 40 x 256
        # ====== SHORT FILTERS ======
        self.sf_conv1 = nn.Conv2d(1, 16, (1,5)) # 16x36x252
        self.sf_conv1_bn = nn.BatchNorm2d(1)
        self.sf_conv2 = nn.Conv2d(16, 16, (1,5)) # 16x32x248
        self.sf_conv2_bn = nn.BatchNorm2d(16)
        self.sf_conv3 = nn.Conv2d(16, 16, (1,5)) # 16x28x244
        self.sf_conv3_bn = nn.BatchNorm2d(16)

        # Multi filter NN x4
        # ====== MULTI FILTER MODS ======
        m=5
        input_chans=16
        self.mf1_ap = nn.AvgPool2d((m,1))
        self.mf1_bn = nn.BatchNorm2d(input_chans)
        self.mf1_conv1 = nn.Conv2d(input_chans,24,(1,32))
        self.mf1_conv2 = nn.Conv2d(input_chans,24,(1,64))
        self.mf1_conv3 = nn.Conv2d(input_chans,24,(1,96))
        self.mf1_conv4 = nn.Conv2d(input_chans,24,(1,128))
        self.mf1_conv5 = nn.Conv2d(input_chans,24,(1,192))
        self.mf1_conv6 = nn.Conv2d(input_chans,24,(1,244)) # This is a single fullscale, cutoff by previous CNNs
        self.mf1_conv_final = nn.Conv2d(24,36,1)

        m=2
        input_chans=36
        self.mf2_ap = nn.AvgPool2d((m,1))
        self.mf2_bn = nn.BatchNorm2d(input_chans)
        self.mf2_conv1 = nn.Conv2d(input_chans,24,(1,32))
        self.mf2_conv2 = nn.Conv2d(input_chans,24,(1,64))
        self.mf2_conv3 = nn.Conv2d(input_chans,24,(1,96))
        self.mf2_conv4 = nn.Conv2d(input_chans,24,(1,128))
        self.mf2_conv5 = nn.Conv2d(input_chans,24,(1,192))
        self.mf2_conv6 = nn.Conv2d(input_chans,24,(1,244)) # This is a single fullscale, cutoff by previous CNNs
        self.mf2_conv_final = nn.Conv2d(24,36,1)

        self.mf3_ap = nn.AvgPool2d((m,1))
        self.mf3_bn = nn.BatchNorm2d(input_chans)
        self.mf3_conv1 = nn.Conv2d(input_chans,24,(1,32))
        self.mf3_conv2 = nn.Conv2d(input_chans,24,(1,64))
        self.mf3_conv3 = nn.Conv2d(input_chans,24,(1,96))
        self.mf3_conv4 = nn.Conv2d(input_chans,24,(1,128))
        self.mf3_conv5 = nn.Conv2d(input_chans,24,(1,192))
        self.mf3_conv6 = nn.Conv2d(input_chans,24,(1,244)) # This is a single fullscale, cutoff by previous CNNs
        self.mf3_conv_final = nn.Conv2d(24,36,1)
        #
        # # GPU NOT ENOUGH RAM
        # self.mf4_ap = nn.AvgPool2d((m,1))
        # self.mf4_bn = nn.BatchNorm2d(input_chans)
        # self.mf4_conv1 = nn.Conv2d(input_chans,24,(1,32))
        # self.mf4_conv2 = nn.Conv2d(input_chans,24,(1,64))
        # self.mf4_conv3 = nn.Conv2d(input_chans,24,(1,96))
        # self.mf4_conv4 = nn.Conv2d(input_chans,24,(1,128))
        # self.mf4_conv5 = nn.Conv2d(input_chans,24,(1,192))
        # self.mf4_conv6 = nn.Conv2d(input_chans,24,(1,244)) # This is a single fullscale, cutoff by previous CNNs
        # self.mf4_conv_final = nn.Conv2d(24,36,1)

        # ====== DENSE LAYERS ======
        # Dense Layers
        # mfmod_size = 36*40908
        mfmod_size = 205632 #1mod
        # mfmod_size=508896
        mfmod_size = 1472688 #3mod
        # mfmod_size = 4391064 #4mod
        self.dl_bn1 = nn.BatchNorm1d(mfmod_size)
        self.dl_do = nn.Dropout(0.5)
        self.dl_fc1 = nn.Linear(mfmod_size, 64)
        self.dl_bn2 = nn.BatchNorm1d(64)
        self.dl_fc2 = nn.Linear(64,64)
        self.dl_bn3 = nn.BatchNorm1d(64)
        self.dl_fc3 = nn.Linear(64,256)


    def forward(self, x):
        # ====== SHORT FILTERS ======
        x = F.elu(self.sf_conv1(self.sf_conv1_bn(x)))
        x = F.elu(self.sf_conv2(self.sf_conv2_bn(x)))
        x = F.elu(self.sf_conv3(self.sf_conv3_bn(x)))

        # ====== MULTI FILTER MODS ======
        x = self.mf1_ap(x)
        x = self.mf1_bn(x)
        c1 = self.mf1_conv1(x)
        c2 = self.mf1_conv2(x)
        c3 = self.mf1_conv3(x)
        c4 = self.mf1_conv4(x)
        c5 = self.mf1_conv5(x)
        c6 = self.mf1_conv6(x)
        x = torch.cat((c1,c2,c3,c4,c5,c6), dim=3)
        x = self.mf1_conv_final(x)

        # SPEED UP BEGINNING TRAINGIN
        x = self.mf2_ap(x)
        x = self.mf2_bn(x)
        c1 = self.mf2_conv1(x)
        c2 = self.mf2_conv2(x)
        c3 = self.mf2_conv3(x)
        c4 = self.mf2_conv4(x)
        c5 = self.mf2_conv5(x)
        c6 = self.mf2_conv6(x)
        x = torch.cat((c1,c2,c3,c4,c5,c6), dim=3)
        x = self.mf2_conv_final(x)

        x = self.mf3_ap(x)
        x = self.mf3_bn(x)
        c1 = self.mf3_conv1(x)
        c2 = self.mf3_conv2(x)
        c3 = self.mf3_conv3(x)
        c4 = self.mf3_conv4(x)
        c5 = self.mf3_conv5(x)
        c6 = self.mf3_conv6(x)
        x = torch.cat((c1,c2,c3,c4,c5,c6), dim=3)
        x = self.mf3_conv_final(x)
        #
        # # GPU not enough RAM
        # x = self.mf4_ap(x)
        # x = self.mf4_bn(x)
        # c1 = self.mf4_conv1(x)
        # c2 = self.mf4_conv2(x)
        # c3 = self.mf4_conv3(x)
        # c4 = self.mf4_conv4(x)
        # c5 = self.mf4_conv5(x)
        # c6 = self.mf4_conv6(x)
        # x = torch.cat((c1,c2,c3,c4,c5,c6), dim=3)
        # x = self.mf4_conv_final(x)

        # ====== DENSE LAYERS ======
        x = torch.flatten(x,1)
        x = self.dl_do(x)
        x = self.dl_bn1(x)
        x = F.elu(self.dl_fc1(x))
        x = self.dl_bn2(x)
        x = F.elu(self.dl_fc2(x))
        x = self.dl_bn3(x)
        x = self.dl_fc3(x)

        # output = F.softmax(x,dim=1)
        output = F.log_softmax(x,dim=1)
        # print('output', output.shape)
        return output
