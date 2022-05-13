import torch.nn as nn
import torch

class ClsNet(nn.Module):
    def __init__(self, num_class=21):
        super(ClsNet, self).__init__()

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

        self.pre_fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_class)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        #print(x.shape)
        x = self.pool1(x)
        #print(x.shape)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        #print(x.shape)
        x = self.pool5(x)
        #print(x.shape)

        x = self.avg_pool(x)
        #print(x.shape)
        x = torch.flatten(x, 1)
        #print(x.shape)
        logits = self.pre_fc(x)
        #print(logits.shape)
        #print(logits)
        prob = self.softmax(logits)
        #print(prob.shape)
        #print(prob)

        return logits, prob

if __name__ == '__main__':
    cls_net = ClsNet()
    img = torch.rand([4, 3, 256, 256])
    logits, prob = cls_net(img)
    print(logits.shape)
    print(prob.shape)
