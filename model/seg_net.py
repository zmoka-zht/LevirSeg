import torch
import torch.nn as nn

class DownConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):

        x = self.conv(x)
        x = self.pool(x)

        return x

class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpBlock, self).__init__()

        self.UpConv = nn.Conv2d(in_channel, out_channel, 3, 1, padding=1)
        self.UpSample = nn.UpsamplingBilinear2d(scale_factor = 2)

    def forward(self, x_before, x_after):
        x_after = self.UpSample(x_after)
        x_after = self.UpConv(x_after)
        x = x_before + x_after

        return x



class SegNet(nn.Module):
    def __init__(self, num_class=6):
        super(SegNet, self).__init__()

        self.block1 = DownConv(3, 32) #b, 32, 128, 128
        self.block2 = DownConv(32, 64) #b, 64, 64, 64
        self.block3 = DownConv(64, 128) #b, 128, 32, 32
        self.block4 = DownConv(128, 256) #b, 256, 16, 16
        self.block5 = DownConv(256, 512) #b, 512, 8, 8

        self.up5 = UpBlock(512, 256)
        self.up4 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.up2 = UpBlock(64, 32)
        self.up1 = nn.Upsample(scale_factor=2)

        self.seghead = nn.Conv2d(32, num_class, 1, 1)

        self.softmax = nn.Softmax(dim=1)





    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        x4_ = self.up5(x4, x5)
        x3_ = self.up4(x3, x4_)
        x2_ = self.up3(x2, x3_)
        x1_ = self.up2(x1, x2_)
        x = self.up1(x1_)

        logits = self.seghead(x)
        prob = self.softmax(logits)

        return logits, prob

if __name__ == '__main__':
    seg_net = SegNet()
    img = torch.rand([4, 3, 256, 256])
    logits, prob = seg_net(img)
    print(logits.shape)