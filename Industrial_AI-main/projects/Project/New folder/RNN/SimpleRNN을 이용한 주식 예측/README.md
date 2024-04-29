# SimpleRNN을 이용한 주식 예측
RNN은 순환 신경망(Recurrent Neural Networks)을 의미합니다. RNN은 순차 데이터가 있을 때 사용됩니다.
`SimpleRNN`은 `Tensorflow`, `Keras`에서 제공하는 함수입니다. 출력이 입력으로 피드백되는 완전 연결형 RNN입니다.
RNN은 **시계열 예측**에 사용됩니다.

## 프로젝트 목표
프로젝트의 목표는 **Tesla Stocks의 성과를 예측**하기 위해 RNN 모델을 실행하는 것입니다.

<img src="https://github.com/navi1910/StockPrediction-SimpleRNN-TimeSeriesForecasting/blob/master/tesla_logo.png" 너비=30% 높이=30%>

## 방법
* 딥러닝 신경망
* 'Tensorflow'의 간단한 RNN
* 시각화
* 기능 엔지니어링
* 데이터 정규화
* `os`를 사용하여 모델 저장

## 사용된 기술
* 파이썬
* 팬더
* 텐서플로우
* 케라스
* 사이킷런(Scikit-learn)
* 맷플롯립

## 프로젝트 설명
데이터는 **Kaggle**에서 얻었습니다. 이 프로젝트는 Tesla 주식의 움직임을 예측하는 것을 목표로 합니다.
순환 신경망에는 **동일한 인공 뉴런의 출력을 다시 입력으로** 가져오는 기능이 있습니다. 신경망은 처음 50개의 관측치를 가져온 다음 회귀 분석을 사용하여 51번째 관측치를 예측합니다.
52번째 관찰 등을 예측하는 프로세스가 계속됩니다. 이 방법을 **자동회귀**라고도 합니다.

RNN 모델의 출력-입력 루프 기능과 Autoregression 프로세스는 미래 값을 예측하는 데 도움이 됩니다.

## 절차
* 필수 Python 모듈을 가져옵니다.
* 데이터는 **Pandas를 사용하여 데이터 프레임**으로 가져옵니다.
* 데이터는 **Training set과 Validation set**으로 구분됩니다.
* 훈련 세트와 검증 세트 모두의 데이터가 재구성되었습니다.
* 데이터는 Scikitlearn의 'MinMaxScaler'를 사용하여 정규화됩니다.
* **특성 추출**은 모델의 요구 사항에 따라 수행됩니다. 50개의 관측치를 사용하여 51번째 관측치를 예측하기 위해 데이터가 분리됩니다.
* 데이터가 다시 재구성됩니다.

### 모델 구축
* 'Tensorflow', 'Keras'는 신경망 구축에 사용됩니다.
* `SimpleRNN`, `Dropout`, `Dense` 레이어가 `Sequential`에 추가됩니다.
* 모델은 다음을 사용하여 '컴파일'됩니다.
     + ``adam'' 최적화 프로그램
     - ``mean_squared_error'' 손실
     + `['정확도']` 측정항목

* 모델이 요약됩니다.

<img src="https://github.com/navi1910/StockPrediction-SimpleRNN-TimeSeriesForecasting/blob/master/model_summary.png" 너비=50% 높이=50%>

* 모델은 훈련 데이터에 적합합니다.
* 손실과 정확도가 표시됩니다.
* X_train은 `predict`를 사용하여 예측한 다음 `inverse_transform`을 사용하여 비정규화됩니다.
* 예측값은 실제값과 함께 표시됩니다.
![기차 도표](https://github.com/navi1910/StockPrediction-SimpleRNN-TimeSeriesForecasting/blob/master/train_prediction.png "기차 도표")
* 검증 세트에 대해 프로세스가 반복되고 결과가 플롯됩니다.
![검증 도표](https://github.com/navi1910/StockPrediction-SimpleRNN-TimeSeriesForecasting/blob/master/validation_prediction.png "검증 도표")

## 모델 저장
모델은 `model.save` 및 Python `os`를 사용하여 `model`이라는 디렉터리에 h5 파일로 저장됩니다. 모델은 `load_model`을 사용하여 로드할 수 있습니다.


## 감사의 말씀
[캐글 노트](https://www.kaggle.com/code/ozkanozturk/stock-price-prediction-by-simple-rnn-and-lstm)