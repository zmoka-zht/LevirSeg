import torch
import torch.nn as nn
from torchvision import models


class ResNet50FCN(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet50FCN, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2)

        self.conv_fpn1 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.conv_fpn2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv_fpn3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv_fpn4 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.conv_pred_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_pred_2 = nn.Conv2d(64, 6, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        # resnet layers
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128
        x_16 = self.resnet.layer3(x_8) # 1/16, in=128, out=256
        x_32 = self.resnet.layer4(x_16) # 1/32, in=256, out=512

        # FPN layers
        x = self.upsample(self.relu(self.conv_fpn1(x_32)))
        x = self.upsample(self.relu(self.conv_fpn2(x + x_16)))
        x = self.upsample(self.relu(self.conv_fpn3(x + x_8)))
        x = self.upsample(self.relu(self.conv_fpn4(x + x_4)))

        # output layers
        x = self.upsample(self.relu(self.conv_pred_1(x)))
        logits = self.conv_pred_2(x)
        prob = self.sigmoid(self.conv_pred_2(x))

        return logits, prob

if __name__ == '__main__':

    model = ResNet50FCN()

    img = torch.rand([4, 3, 256, 256])
    out1, out2 = model(img)

    print(out1.shape)
    print(out2.shape)