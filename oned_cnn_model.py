# coding: utf-8
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from torchsummary import summary


class OneDCNN(nn.Module):
    def __init__(self):
        super(OneDCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=1, kernel_size=32, padding=0),
            # nn.Sigmoid(),
            # nn.MaxPool1d(3, stride=1),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=128, padding=0),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=256, padding=0),
            nn.Sigmoid(),
            nn.MaxPool1d(3),
            nn.Dropout(),
        )
        self.classification = nn.Sequential(
            nn.Linear(395, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, input):
        output = self.features(input)
        flaten = torch.flatten(output)
        output = self.classification(flaten)
        return output


if __name__ == "__main__":
    df = pd.read_csv(f"No1_01.csv", header=None)
    x1 = df[0][:1600].astype('double').to_numpy()
    y1 = df[1][:1600].astype('double').to_numpy()
    z1 = df[2][:1600].astype('double').to_numpy()
    test = torch.from_numpy(np.array([[x1, y1, z1]])).float()
    # test = torch.from_numpy(np.array(x1)).float()

    # test = test.unsqueeze(0).unsqueeze(0)
    print(test.size())
    cnn = OneDCNN()
    print(cnn)

    cnn(test)
    # cnn1d_1 = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=64)
    # cnn1d_2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=64)
    # print(cnn1d_1(test).shape, "\n")
    # print(cnn1d_1(test).size())

    exit()
