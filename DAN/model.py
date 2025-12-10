import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DANNet(nn.Module):
    def __init__(self, num_classes=4):
        super(DANNet, self).__init__()

        pretrained_alexnet = models.alexnet(pretrained=True)

        self.feature_extractor = pretrained_alexnet.features

        self.flatten = nn.Flatten()
        self.flatten_dim = 256 * 6 * 6

        self.fc6 = nn.Linear(self.flatten_dim, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)

        fc6 = self.dropout(F.relu(self.fc6(x)))
        fc7 = self.dropout(F.relu(self.fc7(fc6)))
        output = self.fc8(fc7)

        return output, fc6, fc7, self.fc8(fc7)

