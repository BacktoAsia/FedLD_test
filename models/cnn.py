import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, Bottleneck


class CifarCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CifarCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes, bias=True)
        self.base_weight_keys = [
                                'conv1.weight', 'conv1.bias',
                                'conv2.weight', 'conv2.bias',
                                'conv3.weight', 'conv3.bias',
                                'fc1.weight', 'fc1.bias',
                                ]
        self.classifier_weight_keys = [
                                'fc2.weight', 'fc2.bias',
                                ]

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.leaky_relu(self.fc1(x))
        y = self.fc2(x)
        return x, y
    
    def feature2logit(self, x):
        return self.fc2(x)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CNN_FMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_FMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=1)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes, bias=True)
        self.base_weight_keys = ['conv1.weight', 'conv1.bias',
                            'conv2.weight', 'conv2.bias',
                            'fc1.weight', 'fc1.bias',]
        self.classifier_weight_keys = ['fc2.weight', 'fc2.bias',]

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.leaky_relu(self.fc1(x))
        y = self.fc2(x)
        return x, y

    def feature2logit(self, x):
        return self.fc2(x)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class FC_SYN(nn.Module):
    def __init__(self, input_dim=60, output_dim=10):
        super(FC_SYN, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.classifier_weight_keys = ['fc.weight', 'fc.bias']

    def forward(self, x):
        y = self.fc(x)
        return x, y


class ResNet50(ResNet):
    def __init__(self):
        super(ResNet50, self).__init__(Bottleneck, [3, 4, 6, 3])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.fc(x)

        return x,y