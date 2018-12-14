import torch
import torch.nn as nn

# Resnet block
class ResnetBlock(nn.Module):
    def __init__(self, inDim, outDim):
        super(ResnetBlock, self).__init__()
        self.layer = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(inDim, outDim, 3, 1, 0),
                                   nn.BatchNorm2d(outDim),
                                   nn.ReLU(),
                                   nn.ReflectionPad2d(1),
                                   nn.Conv2d(outDim, outDim, 3, 1),
                                   nn.BatchNorm2d(outDim))

    def forward(self, m):
        # a_t + a_t+2 = m + layer(m) 
        return self.layer(m) + m 

# Generator network (nResnets+6 layers)
class Generator(nn.Module):
    def __init__(self,nResnets):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

        # Add ResNet blocks
        resnets = []
        for i in range(1,nResnets):
            resnets.append(ResnetBlock(256, 256))
        self.layer4 = nn.Sequential(*resnets)

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, 
                                         padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(128, 64,  kernel_size=3, stride=2, 
                                         padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=0),
            nn.Tanh())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        return out

# Discriminator network (5 layers)
class Discriminator_5(nn.Module):
    def __init__(self):
        super(Discriminator_5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=(2,1)),
            nn.LeakyReLU(negative_slope=0.2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=(2, 1)))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out

