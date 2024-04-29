# CNN
- Convolutional Neural Networks
- 수동으로 특징을 추출할 필요 없이 데이터로부터 직접 학습하는 딥러닝을 위한 신경망 아키텍처
- 이미지의 특징을 추출하는 부분과 클래스를 분류하는 부분으로 나뉜다
- 특징 추출 영역: 합성곱층(Convolution layer)과 풀링층(Pooling lyaer)을 여러겹 쌓는 형태(Conv + Maxpool)로 구성
![image](https://github.com/ChaesongYun/CNN/assets/139418987/fac39790-dd99-44fd-bf2d-c34f7e8419d6)

  이미지의 클래스를 분류하는 부분은 Fully connected(FC) 학습 방식으로 이미지를 분류한다
- 영상에서 객체, 얼굴, 장면 인식을 위한 패턴을 찾을 때 특히 유용
- 오디오, 시계열, 신호 데이터와 같이 영상 이외에 데이터를 분류하는 데도 효과적
- 자율주행차량, 얼굴 인식 응용 분야와 같이 객체 인식과 컴퓨터 비전이 필요한 응용 분야에서도 많이 사용된다!
<br>
<br>

## 표의 열 vs 포함관계
![image](https://github.com/ChaesongYun/CNN/assets/139418987/fd8bb1a5-50b9-40d4-9294-e8a7d170a702)
![image](https://github.com/ChaesongYun/CNN/assets/139418987/4e19369e-9f82-4ff9-964d-a71332d3ece9)

### 배열의 깊이 = "차원수"

![image](https://github.com/ChaesongYun/CNN/assets/139418987/a887d344-7757-4e2f-8dd0-7bbad4680870)
![image](https://github.com/ChaesongYun/CNN/assets/139418987/3cb3f7bf-8325-47cc-98a4-828c750d0935)


- 데이터 공간의 맥락: 차원수 = 변수의 개수
- 데이터 형태의 맥락: 차원수 = 배열의 깊이
<br>
<br>

![image](https://github.com/ChaesongYun/CNN/assets/139418987/e1364d24-7eea-4b09-b769-ff35505886e5)
<br>

개별 Mnist - (28, 28)
- 이 이미지는 2차원 형태이고
- 784(28*28)차원 공간의 한 점이다
<br>
<br>

![image](https://github.com/ChaesongYun/CNN/assets/139418987/feba9f35-67be-4ee5-b55f-78bbe18f24ab)
<br>
개별 cifar10 - (32, 32, 3)
- 이 이미지는 3차원 형태이고
- 3032(32*32*3)차원 공간의 한 점이다
- 만약 이런 이미지가 5만장이 있다면 cifar10 셋은 (50000, 32, 32, 3)이 되는거

![image](https://github.com/ChaesongYun/CNN/assets/139418987/4be9c9bd-b7d1-4698-9adf-22c58564cb63)
