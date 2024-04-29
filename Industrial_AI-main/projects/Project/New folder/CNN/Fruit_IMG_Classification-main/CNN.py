"""
CNN 구조 및 Train, evaluate 함수 출처
- 이경택 외 2명, 『파이썬 딥러닝 파이토치』, 정보문화사(2020), p135-136, p141-145.
- Horea MureÈZan and Mihai Oltean. Fruit recognition from images using deep learning. Acta Universitatis Sapientiae, Informatica, 10:26–42, 06 2018.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# GPU환경이 가능하면 GPU를 사용하도록 설정
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device :', DEVICE)

'''
실험을 진행하였을 때에는 수동으로 일일이 Layer의 채널수 같은 것들을 직접 손으로 수정하여 실험을 진행하였다.
모델에 대한 코드 설명은 보고서에 상세히 적어 놓았다.
'''


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=5,
            padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            padding=1)
        '''
        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=5,
            padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=5,
            padding=1)
        '''
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2)
        self.fc1 = nn.Linear(6400, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 131)
        self.dropout = 0

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        '''
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool(x)
        X = self.conv5(x)
        x = F.relu(x)
        x = self.pool(x)
        '''

        x = x.view(-1, 6400)
        x = self.fc1(x)
        x = F.relu(x)
        if self.dropout > 0:
            x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.fc2(x)
        x = F.relu(x)
        if self.dropout > 0:
            x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


def train(model, train_loader, optimizer, log_interval, criterion, epoch):
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        if batch_idx == len(train_loader) - 1:
            last_loss = loss

        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}".format(
                epoch, batch_idx * len(image),
                len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))

    return last_loss.cpu().detach().numpy()


def evaluate(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100 * correct / len(test_loader.dataset)
    return test_loss, test_accuracy
