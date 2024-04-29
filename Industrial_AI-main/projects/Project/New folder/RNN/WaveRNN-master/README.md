# WaveRNN

#####(업데이트: Vanilla Tacotron One TTS 시스템이 구현. 더 많은 시스템이 곧 출시될 예정!)

![WaveRNN 다이어그램이 포함된 Tacotron](assets/tacotron_wavernn.png)

[효율적인 신경 오디오 합성](https://arxiv.org/abs/1802.08435v1)에서 Deepmind의 WaveRNN 모델을 Pytorch로 구현

# 설치

다음 사항을 확인하세요.

* 파이썬 >= 3.6
* [CUDA를 사용한 Pytorch 1](https://pytorch.org/)

그런 다음 pip를 사용하여 나머지를 설치합니다.

> pip 설치 -r 요구사항.txt

# 사용하는 방법

### 빠른 시작

TTS 기능을 즉시 사용하려면 다음을 사용하면 됩니다.

> 파이썬 Quick_start.py

그러면 기본 문장.txt 파일의 모든 내용이 생성되고 wav 파일을 재생하고 주의 플롯을 살펴볼 수 있는 새로운 'quick_start' 폴더로 출력됩니다.

해당 스크립트를 사용하여 사용자 정의 tts 문장을 생성하거나 '-u'를 사용하여 일괄 처리되지 않은(더 나은 오디오 품질) 생성할 수도 있습니다.

> pythonquick_start.py -u --input_text "이 명령을 실행하면 어떻게 되나요?"


### 자신만의 모델 훈련하기
![집중 및 멜 트레이닝 GIF](assets/training_viz.gif)

[LJSpeech](https://keithito.com/LJ-Speech-Dataset/) 데이터 세트를 다운로드하세요.

**hparams.py**를 편집하고 **wav_path**가 데이터세트를 가리키도록 한 후 다음을 실행하세요.

> 파이썬 preprocess.py

또는 preprocess.py --path를 사용하여 데이터 세트를 직접 가리킵니다.
___

작업 실행 순서에 대한 권장 사항은 다음과 같습니다.

1 - 다음을 사용하여 Tacotron을 훈련합니다.

> 파이썬 train_tacotron.py

2 - 교육을 마치고 나가거나 언제든지 다음을 사용할 수 있습니다.

> 파이썬 train_tacotron.py --force_gta

이렇게 하면 Tactron이 훈련이 완료되지 않은 경우에도 GTA 데이터 세트를 생성하게 됩니다.

3 - 다음을 사용하여 WaveRNN 교육:

> 파이썬 train_wavernn.py --gta

주의: TTS에 관심이 없다면 언제든지 --gta 없이 train_wavernn.py를 실행할 수 있습니다.

4 - 다음을 사용하여 두 모델로 문장을 생성합니다.

> 파이썬 gen_tacotron.py wavernn

그러면 기본 문장이 생성됩니다. 사용자 정의 문장을 생성하려면 다음을 사용할 수 있습니다.

> python gen_tacotron.py --input_text "이것은 당신이 원하는 대로 됩니다" wavernn

마지막으로, 언제든지 해당 스크립트에서 --help를 사용하여 사용 가능한 옵션을 확인할 수 있습니다. :)



# 견본

[여기에서 확인하실 수 있습니다.](https://fatchord.github.io/model_outputs/)

# 사전 훈련된 모델

현재 /pretrained/ 폴더에는 두 가지 사전 훈련된 모델이 있습니다.

둘 다 LJSpeech에 대한 교육을 받았습니다.

* 800,000단계로 훈련된 WaveRNN(물류 혼합 출력)
* Tacotron은 180,000단계까지 훈련되었습니다.

____

### 참고자료

* [효율적인 신경 오디오 합성](https://arxiv.org/abs/1802.08435v1)
* [Tacotron: 엔드투엔드 음성 합성을 향하여](https://arxiv.org/abs/1703.10135)
* [Mel 스펙트로그램 예측에서 WaveNet을 조절하여 자연 TTS 합성](https://arxiv.org/abs/1712.05884)

### 감사의 말씀

* [https://github.com/keithito/tacotron](https://github.com/keithito/tacotron)
* [https://github.com/r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)
* github 유저인 [G-Wang](https://github.com/G-Wang), [geneing](https://github.com/geneing) & [erogol](https://github. com/erogol)