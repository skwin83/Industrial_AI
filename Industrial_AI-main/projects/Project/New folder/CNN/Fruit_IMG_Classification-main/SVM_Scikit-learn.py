"""
코드 구조 출처 : https://ichi.pro/ko/thundersvm-sogae-gpu-mich-cpuui-ppaleun-svm-laibeuleoli-244228560973701
기존엔 GPU를 사용하는 thunderSVM을 사용하려고 했으나 라이브러리 import 문제가 생겨 그냥 svm으로 실행하였음.
SVM 출처 : https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
전체 데이터셋을 하는 것은 시간이 너무 오래 걸려서 일부 클래스(18개의 클래스)만 이용하여 진행하였다.
"""
from sklearn.svm import SVC
import numpy as np


# Loading Training and Test Data
train_x = np.load('train_X.npy')
train_y = np.load('train_y.npy')
test_x = np.load('test_X.npy')
test_y = np.load('test_y.npy')

# Flatten ( 100 * 100 * 3 to 30000 )
print("Now Flatten")
train_x = train_x[:8316].reshape(-1, 30000).astype("float32") / 255
test_x = test_x[:2777].reshape(-1, 30000).astype("float32") / 255


'''
Initialize model

Set Hyperparameter (다양하게 존재하나 일부만 소개한다.)
C : default 값은 1이며, 값이 커지면 Hard Margin(엄격한 기준), 작으면 Soft Margin(너그러운 기준) 이다.
kernel : 커널의 종류를 설정한다. {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}가 있고, default는’rbf’ 이다.
gamma : 하나의 데이터 샘플이 영향력을 행사하는 거리(가중치)를 결정한다. {'scale', 'auto'}가 존재한다.
default는 'scale' 이고 1 / (n_features * X.var()) 이 적용되고, 'auto' 는 1 / n_features을 사용한다.
'''
model = SVC(gamma='auto', kernel='rbf')

print("Model fit")
# Fit the training data to Model
model.fit(train_x[:8316], train_y[:8316])
print("Model Predict")
# Check test set accuracy
accuracy = model.score(test_x[:2777], test_y[:2777])

print('Accuracy: {}'.format(accuracy))
