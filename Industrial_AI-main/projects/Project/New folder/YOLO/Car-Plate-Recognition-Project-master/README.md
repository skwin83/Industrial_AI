# Car-Plate-Recognition-Project
딥러닝을 활용한 자동차 번호판 인식을 위한 엔드투엔드 파이프라인

![파이프라인 개요](/imgs/INFO7374%20CarPlate%20Pipeline%20v2.0.png)

### 내용물
  1. 번호판 감지를 위한 YoloV3
  2. 이미지 전처리 기술을 사용한 숫자 분할
  3. 분할된 숫자를 분류하기 위해 훈련된 CNN 분류기
  4. 감지된 판의 텍스트를 예측하기 위한 CRNN 광학 문자 인식 모델을 위한 훈련 파이프라인(모델은 효율성을 높이기 위해 더 많은 데이터에 대해 훈련되어야 함)
  5. 튜토리얼 노트
  6. YOLO와 분할, CNN 분류의 조합을 이용한 추론을 위한 플라스크 적용
  7. 사전 훈련된 SSD 모델 aws 인공물을 사용한 AWS Deeplens 자습서
  8. 공기 흐름 훈련 파이프라인
           
            - CNN 훈련 파이프라인
            - CRNN 훈련 파이프라인
  9. 공기 흐름 파이프라인 도커화
            
             Docker 서비스를 실행하려면 이 저장소를 복제하고 이 폴더에 CD를 넣으세요.
             - docker-compose -f docker-compose-CeleryExecutor.yml up
             - docker-compose -f docker-compose-CeleryExecutor.yml up -d (분리 모드)
 
#### YOLOv3를 사용한 객체 감지:
      객체 감지에 대한 이전 작업에서는 감지를 수행하기 위해 분류자의 용도를 변경했습니다. 대신, 객체 감지는 공간적으로 분리된 경계 상자 및 관련 클래스 확률에 대한 회귀 문제로 구성됩니다.
      단일 신경망은 한 번의 평가로 전체 이미지에서 직접 경계 상자와 클래스 확률을 예측합니다. 전체 탐지 파이프라인은 단일 네트워크이므로 탐지 성능에 직접적으로 엔드 투 엔드를 최적화할 수 있습니다.
 
#### AWS DeepLens
      AWS DeepLens는 딥 러닝 지원 비디오 카메라입니다. Amazon Machine과 통합되어 있습니다.
      생태계를 학습하고 AWS 클라우드에서 프로비저닝된 배포 모델에 대해 로컬 추론을 수행할 수 있습니다.
     
##### 지원되는 모델링 프레임워크:
       AWS DeepLens를 사용하면 지원되는 딥 러닝 모델링 프레임워크를 사용하여 프로젝트 모델을 훈련할 수 있습니다.
       AWS 클라우드나 다른 곳에서 모델을 훈련할 수 있습니다. 현재 AWS DeepLens는 Caffe, TensorFlow 및 Apache MXNet 프레임워크를 지원합니다.
      
##### Flask 앱 - 샘플 예측

![이미지](https://user-images.githubusercontent.com/37238004/70835944-a65f0380-1dcc-11ea-8def-d4bda672fbf8.png)
      
      
#### 신고 링크
https://docs.google.com/document/d/1Sa3jxZ_6bPCQz6wDBn-h-yO2jaeonfrUAkS2-ZWV7hU/edit?usp=sharing