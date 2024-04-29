# Fruit Image Classification

## 개요

`CNN` `DNN` `SVM` 을 이용하여 과일 이미지 분류 모델을 구현하고 각각의 결과를 분석한다

## 참고 자료

`Keras 공식 Document` 에서 제공하는 SVM 을 이용하는 KNN 예제를 참고하여 구현했다. <br>
이후에 내가 구현한 것과 비교를 위해 `Scikit-learn` 의 SVM 을 참고했다. <br>

## 구현환경

- IDE : VS Code, Framework : Keras, Scikit-Learn
- Intel(R) Core(TM) i9-9880H CPU @2.30GHz, RAM 16.0GB, macOS Big Sur, Radeon Pro 560X 4GB 환경에서 실행

## SVM Algorithm

`서포트 벡터 머신(support vector machine, SVM)`은 기계 학습의 분야 중 하나로 패턴 인 식, 자료 분석을 위한 지도 학습 모델이며, 주로 분류와 회귀 분석을 위해 사용한다. <br>
두 카테고리 중 어느 하나에 속한 데이터의 집합이 주어졌을 때, SVM 알고리즘은 주어진 데 이터 집합을 바탕으로 하여 새로운 데이터가 어느 카테고리에 속할지 판단하는 비확률적 이진 선형 분류 모델을 만든다. <br>
만들어진 분류 모델은 데이터가 사상된 공간에서 경계로 표현되는데 SVM 알고리즘은 그 중 가장 큰 폭을 가진 경계를 찾는 알고리즘이다. <br>

![그림1](https://user-images.githubusercontent.com/55660691/125154544-653ff880-e195-11eb-81ab-d8c0c3180323.png)

## 최적의 결정 경계

SVM 이라는 이름에서 Support Vectors 는 결정 경계와 가장 가까이 있는 데이터 포인트들을 의미한다. <br>
이 데이터들이 경계를 정의하는 결정적인 역할을 하는데, 아래 6개의 그래프에서는 F 그래프가 두 클래스(분류) 사이에서 거리가 가장 멀기 때문에 가장 적절하고, 이때 결정 경계는 데이터 군으로부터 최대한 멀리 떨어지는 게 좋다는 것을 확인할 수 있다.<br>

![2](https://user-images.githubusercontent.com/55660691/125154545-65d88f00-e195-11eb-8533-b43d5bcc4af7.png)

## 마진(Margin)

마진(Margin)은 결정 경계와 서포트 벡터 사이의 거리를 의미한다. <br>
아래 그림에서 가운데 실선이 결정 경계이고, 그 실선으로부터 검은 테두리가 있는 빨간점 1개, 파란점 2개까지 영역을 두고 점선이 그어져있는데, 이때 점선으로부터 결정 경계까지의 거리가 마진(Margin)이다.
여기서 최적의 결정 경계는 마진이 최대화된 경우이며, n개의  속성을 가진 데이터에는 최소 n+1개의 서포트 벡터가 존재한다는 것을 확인할 수 있다. <br>
실제로 SVM에서는 결정 경계를 정의하는 서포트 벡터만 잘 골라내면 나머지 불필요한 데이터 포인트들을 무시할 수 있어 매우 빠르다. <br>

![3](https://user-images.githubusercontent.com/55660691/125154546-65d88f00-e195-11eb-8f33-abecd833272c.png)

## 이상치(Outlier)

분류 과정에서 혼자 튀는 데이터 포인터들을 이상치(Outlier)라고 한다. <br>
아래 그림 중 첫번째 그림은 이상치를 허용하지않고 기준을 까다롭게 세운 경우인데, 이는 하드 마진(Hard margin)으로 마진(Margin)이 매우 작아지고 오버피팅(overfitting)의 문제가 발생할 수 있다. <br>
두번째 그림은 이상치를 마진 안에 어느정도 포함시키고 기준을 너그럽게 세운 경우인데, 이는 소프트 마진(Soft Margin)으로 마진(Margin)이 커지고 반대로 언더피팅(underfitting)의 문제가 발생할 수 있다.<br>

![4png](https://user-images.githubusercontent.com/55660691/125154547-66712580-e195-11eb-9466-66b94ffecbcc.png)

## 데이터에 대한 설명

데이터셋의 이름은 `Fruits365` 이다. <br>
총 90483개의 이미지로 이루어져 있고, `Training Set`이 `67692개의 이미지`로 약 75%를 차지하고, `Test Set`이 `22688개의 이미지`로 약 25%를 포함한다. <br>
실제로는 이 두 집합 말고 Multiple fruits라는 여러 개의 과일이 같이 있는 사진이 있는데 훈련이나 테스트시 사용하지 않으므로 제외하면 총 90380개의 이미지를 사용한다.
SVM을 실험할 때에는 직접 데이터 폴더에 접근하여 이미지 데이터를 numpy array로 형식을 바꾸어 .npy 의 파일로 저장을 하였고, <br>
DNN과 CNN 실험을 할 때에는 PyTorch의 ImageFolder라는 method를 사용하여 데이터셋을 Load 하였다.

## Input Feature

이 데이터셋의 이미지 종류는 총 131개, 즉 `131개의 클래스`를 가지고 있으며, 이미지의 크기는 전부 `100 * 100 pixel의 크기`로 이루어져 있다. <br>
그리고 RGB 값을 가지기 때문에 `Input Feature은 100 * 100 * 3의 크기`가 된다 <br>

## Target Output

데이터셋에 총 존재하는 Class의 개수는 `131개`이며, <br>
존재하는 클래스의 종류는 Apples (different varieties: Crimson Snow, Golden, Golden-Red, Granny Smith, Pink Lady, Red, Red Delicious), Apricot, Avocado, Avocado ripe, Banana (Yellow, Red, Lady Finger), Beetroot Red, Blueberry, Cactus fruit, Cantaloupe (2 varieties), Carambula, Cauliflower, Cherry (different varieties, Rainier), Cherry Wax (Yellow, Red, Black), Chestnut, Clementine, Cocos, Corn (with husk), Cucumber (ripened), Dates, Eggplant, Fig, Ginger Root, Granadilla, Grape (Blue, Pink, White (different varieties)), Grapefruit (Pink, White), Guava, Hazelnut, Huckleberry, Kiwi, Kaki, Kohlrabi, Kumsquats, Lemon (normal, Meyer), Lime, Lychee, Mandarine, Mango (Green, Red), Mangostan, Maracuja, Melon Piel de Sapo, Mulberry, Nectarine (Regular, Flat), Nut (Forest, Pecan), Onion (Red, White), Orange, Papaya, Passion fruit, Peach (different varieties), Pepino, Pear (different varieties, Abate, Forelle, Kaiser, Monster, Red, Stone, Williams), Pepper (Red, Green, Orange, Yellow), Physalis (normal, with Husk), Pineapple (normal, Mini), Pitahaya Red, Plum (different varieties), Pomegranate, Pomelo Sweetie, Potato (Red, Sweet, White), Quince, Rambutan, Raspberry, Redcurrant, Salak, Strawberry (normal, Wedge), Tamarillo, Tangelo, Tomato (different varieties, Maroon, Cherry Red, Yellow, not ripened, Heart), Walnut, Watermelon. 가 있다.

## 결과

# Data Normalization

기존의 데이터는 0-255 사이의 값들로 구성되어있어 계산의 효율을 위해 정규화가 요구된다. <br>
정규화는 두가지 방식으로 진행하였다. 먼저 데이터들을 [0,1] 사이의 값으로 조정 후에 학습하고, 이후에는 [-1, 1] 사이의 값으로 조정 후에 학습했다. <br>
학습 결과 [0,1] 사이의 값으로 조정한 경우에 더 높은 정확도를 보였다.

![5](https://user-images.githubusercontent.com/55660691/125154534-5fe2ae00-e195-11eb-8f28-7017c07feb7d.png)
![6](https://user-images.githubusercontent.com/55660691/125154536-61ac7180-e195-11eb-97e1-d550457bb3f8.png)

# Batch-Size

Batch-Size 는 한 step 당 처리하는 데이터의 양으로 32가 적절한 값인지 판단하기 위해 16 으로 학습했다. 그 결과 비슷한 수준의 정확도를 보였다. <br>
이미 충분히 적절한 사이즈의 Batch-size 가 설정되었다고 판단할 수 있었다.

![7](https://user-images.githubusercontent.com/55660691/125154537-62450800-e195-11eb-9528-6ac0e1ef2ac1.png)
![8](https://user-images.githubusercontent.com/55660691/125154538-62dd9e80-e195-11eb-9dde-75dc159a1792.png)

# Learning rate

학습률(Learning rate)은 최소 손실 함수를 향해 이동하면서 각 step에서 그 크기를 결정하는 최적화 알고리즘의 튜닝 매개변수다. <br>
따라서 적절한 학습률 설정을 통해 최저점에 최적의 시간 안에 도달하도록 해야한다. <br>
기존에 0.01에서 0.001로  학습률을 수정했을 때, 정확도가3% 정도 소폭 증가한 것을 확인했다.

![9](https://user-images.githubusercontent.com/55660691/125154539-63763500-e195-11eb-85a3-fc34b8ab5dec.png)
![10](https://user-images.githubusercontent.com/55660691/125154540-640ecb80-e195-11eb-9f54-0fcc93d3da32.png)

# ScikitLearn SVM

마지막으로 Scikit-learn 에서 제공하는 SVM 라이브러리를 이용한 모델과 비교해보았다. <br>
Scikit-learn 에서 직접 제공하는 SVM 라이브러리는 GPU 를 사용할 수 없어 주어진 시간 내에 모델 학습이 불가능했다. <br>
따라서 상위 18개의 클래스에 대한 데이터만 가지고 분류를 실험하였다. 실험 결과 18개의 클래스로 기존 Keras 코드를 이용해 학습했을때 정확도 67%로 소폭 상승했고, <br>
scikit-learn SVM 코드는 98%의 높은 정확도를 보였다.

![11](https://user-images.githubusercontent.com/55660691/125154541-64a76200-e195-11eb-8ba7-a27e8b6abefd.png)
![12](https://user-images.githubusercontent.com/55660691/125154542-64a76200-e195-11eb-8382-0451a305eeed.png)

## 최종 결과

지금까지 총 3개의 모델을 활용하여 다양한 실험을 진행하였다. 각 모델로 실험하여 나온 결과 중 최고로 성능이 좋았던 결과들을 비교하면 다음과 같다.<br>

![스크린샷 2021-07-10 오후 3 30 42](https://user-images.githubusercontent.com/55660691/125154543-653ff880-e195-11eb-8d35-c9c320cde1c0.png)

# 자료 출처

- SVM https://ko.wikipedia.org/wiki/%EC%84%9C%ED%8F%AC%ED%8A%B8_%EB%B2%A1%ED%84%B0_%EB%A8%B8%EC%8B%A0
- 이미지 http://hleecaster.com/ml-svm-concept/



