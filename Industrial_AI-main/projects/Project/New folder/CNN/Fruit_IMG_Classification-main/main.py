"""
ImageFolder : https://hulk89.github.io/pytorch/2017/11/23/pytorch-image-loader/
"""
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from CNN import CNN, DEVICE, train, evaluate
from DNN import DNN

'''
Set HyperParameter
BATCH_SIZE 는 데이터셋을 여러 작은 그룹으로 나누었을 때 하나의 그룹에 속하는 데이터의 수를 의미한다.
EPOCHS 는 Train의 횟수를 의미한다.
log_interval 은 Train 과정에서 중간 과정을 출력을 할 때 그 로그의 간격 이라고 볼수 있다. 작으면 작을수록 중간 결과가 자주 출력된다.
learning_rate는 학습률로, Gradient Descent 과정에서 다음 지점을 결정할 때 사용하는 파라미터이다. 크면 클수록 다음 지점을 더 멀리 선택하게 된다.
Model은 CNN혹은 DNN모델을 선택할 수 있다.
'''
BATCH_SIZE = 32
EPOCHS = 10
log_interval = 200
learning_rate = 0.0001
Model = 'DNN'

if __name__ == "__main__":
    print("___________________________________Start Data Loading___________________________________")
    print('\n\n\n')
    # 폴더형태로 되어있는 데이터셋을 받아오는 코드, 데이터는 평균 0.5, 표준편차 0.5인 데이터로 표준화 되었다.
    dataset = datasets.ImageFolder(root="fruits-360/Training",
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                         ]))

    test_dataset = datasets.ImageFolder(root="fruits-360/Test",
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                         ]))
    # BATCH_DATASET으로 만든다.
    train_loader = DataLoader(dataset=dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             )
    print("___________________________________Finish Data Loading___________________________________")
    print('\n\n\n')
    print("___________________________________Start Training___________________________________")
    print("Your Model : ", Model)
    # DNN혹은 CNN을 선택할 수 있음
    # 역전파 과정에서 파라미터를 업데이트할 때 사용할 Optimizer 설정 (CNN은 Adam, DNN은 SGD)
    if Model == 'CNN':
        model = CNN().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        model = DNN().to(DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Loss는 CrossEntropy로 설정
    criterion = nn.CrossEntropyLoss()

    # Loss와 Accuracy History정보를 담기 위한 Array
    train_loss_history = []
    test_loss_history = []
    test_accuracy_history = []

    # Train 진행
    for epoch in range(1, EPOCHS + 1):
        # Train 진행 후 최종 Loss 값 반환
        train_loss = train(model, train_loader, optimizer, log_interval, criterion, epoch)
        # Test Data로 Evaluate 진행
        test_loss, test_accuracy = evaluate(model, test_loader, criterion)
        # Loss와 Accuracy History를 위한 값 저장
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)
        # 결과 출력
        print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
            epoch, test_loss, test_accuracy))

    # Train Loss와 Test Loss의 History 를 Plotting 후 저장하는 함수
    plt.figure()
    plt.plot(train_loss_history, label='Train')
    plt.plot(test_loss_history, label='Test')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Loss History')
    plt.legend()
    plt.savefig('Loss_History.png')
    plt.show()

    # Test Accuracy의 History를 Plotting 후 저장하는 함수
    plt.clf()
    plt.plot(test_accuracy_history, label='Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Accuracy History')
    plt.savefig('Accuracy_History.png')
    plt.legend()
    plt.show()

