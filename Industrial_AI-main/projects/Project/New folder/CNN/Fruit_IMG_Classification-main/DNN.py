import os
import torch
import torch.nn as nn
import torch.nn.functional as F


path_dir = r"fruits-360"
labels = os.listdir(path_dir)
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

'''
실험을 진행하였을 때에는 수동으로 일일이 Layer의 채널수 같은 것들을 직접 손으로 수정하여 실험을 진행하였다.
모델에 대한 코드 설명은 보고서에 상세히 적어 놓았다.
'''

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(3 * 100 * 100, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 131)
        self.dropout = 0

    def forward(self, x):
        x = x.view(-1, 3 * 100 * 100)
        x = F.relu(self.fc1(x))
        if self.dropout > 0:
            x = F.dropout(x, training=self.training, p=self.dropout)
        x = F.relu(self.fc2(x))
        if self.dropout > 0:
            x = F.dropout(x, training=self.training, p=self.dropout)
        x = F.relu(self.fc3(x))
        if self.dropout > 0:
            x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.fc4(x)
        x = F.log_softmax(x, dim=1)

        return x

