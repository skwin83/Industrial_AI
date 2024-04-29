#YOLO

YOLO("You Only Look Once")는 초당 45프레임에 가까운 실시간 실행이 가능하면서 높은 정확도를 달성하기 때문에 널리 사용되는 알고리즘입니다. 네트워크의 더 작은 버전인 Fast YOLO는 놀라운 초당 155프레임을 처리하는 동시에 다른 실시간 감지기보다 두 배의 mAP를 달성합니다. 이 알고리즘은 예측을 위해 네트워크를 통과하는 순방향 전파가 단 한 번만 필요하다는 의미에서 이미지를 "한 번만 봅니다". 비최대 억제 후 경계 상자와 함께 인식된 객체를 출력합니다.

### 일반적인 YOLO 모델
---------

<img src="https://raw.githubusercontent.com/tejaslodaya/car-Detection-yolo/master/nb_images/model_architecture.png" style="width:500px;height:250;">


### 자동차 감지의 YOLO 모델
------------------

<img src="https://raw.githubusercontent.com/tejaslodaya/car-Detection-yolo/master/nb_images/fig1.png" style="width:500px;height:250;">

- **입력**은 모양(m, 608, 608, 3)의 이미지 배치입니다.
- **출력**은 인식된 클래스와 함께 경계 상자 목록입니다. 각 경계 상자는 위에서 설명한 대로 6개의 숫자 `(p_c, b_x, b_y, b_h, b_w, c)`로 표시됩니다. `c`를 80차원 벡터로 확장하면 각 경계 상자는 85개의 숫자로 표시됩니다.
- 5개의 앵커 박스를 사용하는 경우 YOLO 아키텍처는 다음과 같습니다. IMAGE(m, 608, 608, 3) -> DEEP CNN -> ENCODING(m, 19, 19, 5, 85)
   <img src="https://raw.githubusercontent.com/tejaslodaya/car-Detection-yolo/master/nb_images/fig2.png" style="width:420px;height:240px;">
- 각 셀에는 5개의 상자가 제공됩니다. 전체적으로 모델은 이미지를 한 번만 보면 19x19x5 = 1805개의 상자를 예측합니다(네트워크를 통해 한 번의 전달)! 상자가 너무 많음. 알고리즘의 출력을 훨씬 적은 수의 감지된 개체로 필터링합니다.

### 필터링
-------------
감지된 개체 수를 줄이려면 다음 두 가지 기술을 적용하십시오.
1. 점수 임계값:
   임계값보다 낮은 점수를 가진 클래스를 감지한 상자를 버립니다.
2. 비최대 억제(NMS):
   <img src="https://raw.githubusercontent.com/tejaslodaya/car-Detection-yolo/master/nb_images/non-max-suppression.png" style="width:500px;height:400;">
   이 예에서 모델은 3대의 자동차를 예측했지만 실제로는 동일한 자동차에 대한 3개의 예측입니다. NMS(Non-Max Suppression)를 실행하면 3개의 상자 중 가장 정확한(가장 높은 확률) 상자만 선택됩니다.
  
   NMS를 수행하는 단계는 다음과 같습니다.
   1. 가장 높은 점수를 받은 박스를 선택하세요.
   2. 다른 모든 상자와 겹치는 부분을 계산하고 iou_threshold보다 많이 겹치는 상자를 제거합니다.
   3. 현재 선택한 상자보다 점수가 낮은 상자가 더 이상 없을 때까지 선택한 상자와 겹치는 모든 상자를 반복합니다.
  
예를 들어 box1과 box2라는 두 개의 상자가 있다고 가정합니다.
p(박스1) = 0.9
p(박스2) = 0.6
iou(상자1, 상자2) = 0

따라야 할 단계:
1. 사용 가능한 점수 중 확률이 가장 높은 box1을 선택합니다.
2. 겹치는 부분이 없으므로 box1을 선택합니다.
3. 반복하면 이제 box2의 확률이 가장 높습니다. 상자2를 선택하세요.
4. 루프가 종료됩니다.

더 자세한 논의를 보려면 [문제](https://github.com/tejaslodaya/car-Detection-yolo/issues/1)를 따르세요.

### 결과
-----------
입력 이미지:
   <img src="https://raw.githubusercontent.com/tejaslodaya/car-Detection-yolo/master/nb_images/prediction_input.jpg" style="width:768px;height:432px;">

출력 이미지:
   <img src="https://raw.githubusercontent.com/tejaslodaya/car-Detection-yolo/master/nb_images/prediction_output.jpg" style="width:768px;height:432px;">

### 메모
--------
1. YOLO 모델을 훈련하는 데는 매우 오랜 시간이 걸리며 광범위한 대상 클래스에 대해 레이블이 지정된 경계 상자로 구성된 상당히 큰 데이터 세트가 필요합니다. 이 프로젝트는 공식 YOLO 웹사이트의 기존 사전 훈련된 가중치를 사용하고 Allan Zelener가 작성한 함수를 사용하여 추가 처리됩니다.
2. 전체 모델 아키텍처는 [여기](https://github.com/tejaslodaya/car-Detection-yolo/blob/master/model.png)에서 확인할 수 있습니다.
3. `yolo.h5` 파일을 생성하는 방법은 [여기](https://github.com/allanzelener/YAD2K)에서 확인할 수 있습니다. 'model_data' 폴더에 넣으세요.
4. 입력 이미지는 'images' 디렉터리에서 찾을 수 있습니다. 해당 출력 이미지는 'out' 디렉터리에서 찾을 수 있습니다.

### 참고자료
--------------
1. Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi - 한 번만 보면 됩니다: 통합된 실시간 개체 감지(2015)
2. 조셉 레드먼, 알리 파르하디 - YOLO9000: 더 좋게, 더 빠르게, 더 강하게(2016)
3. Allan Zelener - YAD2K: 또 다른 다크넷 2 Keras
4. YOLO 공식 홈페이지 (https://pjreddie.com/darknet/yolo/)