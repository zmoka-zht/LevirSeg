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

class SegNetz(nn.Module):
    def __init__(self,num_classes=6):
        ##定义网络结构
        super(SegNetz, self).__init__()
        #self.backbone=resnet.resnet50(num_classes=None)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pool1 = nn.AvgPool2d(2, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool2 = nn.AvgPool2d(2, 2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.pool3 = nn.AvgPool2d(2, 2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.pool4 = nn.AvgPool2d(2, 2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.pool5 = nn.AvgPool2d(2, 2)

        self.postConv4 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.postConv3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.postConv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.postConv1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.pred_conv=nn.Conv2d(32,num_classes,1,1,0)

        self.softmax=nn.Softmax(dim=1)

    def forward(self,x):
        #bx 512 x256x256
        x1=self.conv1(x)
        x=self.pool1(x1)
        x2 = self.conv2(x)
        x = self.pool2(x2)
        x3 = self.conv3(x)
        x = self.pool3(x3)
        x4 = self.conv4(x)
        x = self.pool5(x4)
        x5 = self.conv5(x)
        x = self.pool5(x5)
        ##b x 512 x 8 x 8
        #x,endpoints=self.backbone.forward(x)

        x=torch.nn.functional.interpolate(x,scale_factor=2) # b x 512 x 16 x 16
        x=x+x5
        x=self.postConv4(x)     # b x 256 x 16 x 16

        x = torch.nn.functional.interpolate(x, scale_factor=2) # b x 256 x 32 x 32
        x=x+x4
        x=self.postConv3(x) # b x 128 x 32 x 32

        x = torch.nn.functional.interpolate(x, scale_factor=2)# b x 128 x 64 x 64
        x=x+ x3
        x=self.postConv2(x)         # b x 64 x 64 x 64

        x = torch.nn.functional.interpolate(x, scale_factor=2)
        x=x+x2
        x=self.postConv1(x)        # b x 32 x 128 x 128

        x = torch.nn.functional.interpolate(x, scale_factor=2) # b x 32 x img_size x img_size
        x=x+x1
        # x = torch.nn.utils.interpolate(x, size=(img_size,img_size))

        logits=self.pred_conv(x)  # b x num_classes x img_size x img_size
        prob=self.softmax(logits)
        return logits,prob

if __name__ == '__main__':
    seg_net = SegNet()
    img = torch.rand([4, 3, 256, 256])
    logits, prob = seg_net(img)
    print(logits.shape)