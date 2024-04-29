# Simple DQN

**안타깝게도 이 저장소는 오래되었으며 훨씬 더 나은 코드베이스가 있습니다. 기본 사항을 알아보려면 [이것](https://github.com/keon/deep-q-learning)을 살펴보거나 [이것](https://github.com/hill-a/stable)을 살펴보는 것이 좋습니다. -baselines) Atari를 위한 완전한 DQN 구현을 위한 것입니다.**

DeepMind의 결과를 논문 ["심층 강화 학습을 통한 인간 수준 제어"](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)에 복제하기 위한 Deep Q-learning 에이전트. 간단하고 빠르며 쉽게 확장할 수 있도록 설계되었습니다. 특히:
  * 파이썬입니다 :).
  * ALE [네이티브 Python 인터페이스](https://github.com/bbitmaster/ale_python_interface/wiki/Code-Tutorial)를 사용합니다.
  * [OpenAI Gym](https://gym.openai.com/)을 통한 훈련 및 테스트를 지원하도록 업데이트되었습니다.
  * [Neon 딥 러닝 라이브러리](http://neon.nervanasys.com/docs/latest/index.html)의 [가장 빠른 컨볼루션](https://github.com/soumith/convnet-benchmarks).
  * 모든 화면은 재생 메모리에 한 번만 보관되며 Numpy 배열 슬라이싱을 통한 빠른 미니배치 샘플링이 가능합니다.
  * 배열 및 데이터 유형 변환 횟수가 최소화됩니다

Breakout, Pong, Seaquest 및 Space Invaders의 예시 게임플레이 비디오를 확인하세요.

[![Breakout](http://img.youtube.com/vi/KkIf0Ok5GCE/default.jpg)](https://youtu.be/KkIf0Ok5GCE)
[![Pong](http://img.youtube.com/vi/0ZlgrQS3krg/default.jpg)](https://youtu.be/0ZlgrQS3krg)
[![Seaquest](http://img.youtube.com/vi/b6g6A_n8mUk/default.jpg)](https://youtu.be/b6g6A_n8mUk)
[![Space Invaders](http://img.youtube.com/vi/Qvco7ufsX_0/default.jpg)](https://youtu.be/Qvco7ufsX_0)

## Installation

Currently only instructions for Ubuntu are provided. For OS X refer to [ALE](https://github.com/mgbellemare/Arcade-Learning-Environment/blob/master/doc/manual/manual.pdf) and [Neon](http://neon.nervanasys.com/docs/latest/installation.html) documentation.

### Neon

Install prerequisites:
```
sudo apt-get install libhdf5-dev libyaml-dev libopencv-dev pkg-config
sudo apt-get install python python-dev python-pip python-virtualenv
sudo apt-get install libcurl4-openssl-dev
sudo apt-get install libsox-fmt-all libsox-dev sox
```
Check out and compile the code:
```
git clone https://github.com/NervanaSystems/neon.git
cd neon
make
```

필터 시각화를 시험해 보려면 최신 Neon을 사용하고 대신 `make -e VIS=true`를 실행하세요. 시각화 종속성을 활성화하지 않고 이미 Neon을 설치한 경우 `make -e VIS=true` 호출 전에 `vis_requirements.txt`를 터치하여 virtualenv Python 종속성이 트리거되도록 해야 합니다.

Neon은 `.venv`의 가상 환경에 자체적으로 설치됩니다. Python에서 Neon을 가져오려면 활성화해야 합니다.
```
source .venv/bin/activate
```

### Arcade Learning Environment

You can skip this, if you only plan to use OpenAI Gym.

Install prerequisites:
```
sudo apt-get install cmake libsdl1.2-dev
```
Check out and compile the code:
```
git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git
cd Arcade-Learning-Environment
cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON .
make -j 4
```
Install Python library (assuming you have activated Neon virtual environment):
```
pip install .
```

### OpenAI Gym

You can skip this, if you only plan to use Arcade Learning Environment directly.

To install OpenAI Gym:
```
pip install gym
pip install gym[atari]
```

### Simple DQN

Prerequisities:
```
pip install numpy argparse logging
```
Neon 가상 환경에는 이미 해당 라이브러리가 포함되어 있지만 만일을 대비해 여기에 나열되어 있습니다.

또한 가상 환경에 설치하기 어려운 OpenCV가 필요합니다. 나는 이 해킹으로 끝났습니다.
```
sudo apt-get install python-opencv
ln -s /usr/lib/python2.7/dist-packages/cv2.so NEON_HOME/.venv/lib/python2.7/site-packages/
```
`NEON_HOME` here means the folder where you installed (cloned) Neon.

Then just check out the code:
```
git clone https://github.com/tambetm/simple_dqn
cd simple_dqn
```

### Optional

For plotting install `matplotlib`:
```
pip install matplotlib
```

For producing game videos install `avconv`:
```
sudo apt-get install libav-tools
```

## Running the code

### Training

To run training for Breakout:
```
./train.sh roms/breakout.bin
```

If using OpenAI Gym:
```
./train.sh Breakout-v0 --environment gym
```
수많은 옵션이 있습니다. `./train.sh --help`를 실행하여 확인하세요. 훈련하는 동안 네트워크 가중치는 각 에포크 이후 `snapshots` 폴더에 저장됩니다. 파일 이름은 `<game>_<epoch_nr>.pkl`입니다. 훈련 통계는 `results/<game>.csv`에 저장됩니다. 아래에서 플롯을 생성하는 방법을 참조하세요.

### Resuming training

You can resume training by running 
```
./resume.sh snapshots/breakout_10.pkl
```
Pay attention that the replay memory is empty.

### Only testing

To run only testing on a pre-trained model:
```
./test.sh snapshots/breakout_77.pkl
```

To test using OpenAI Gym:
```
./test_gym.sh snapshots/Breakout-v0_77.pkl
```

This saves testing results in folder `results/Breakout-v0`. Now you can then upload your results to OpenAI Gym:
```
./upload_gym.sh results/Breakout-v0 --api_key <your_key>
```
Note that the OpenAI Gym environment differs from the default environment so testing using OpenAI Gym should use a model trained using OpenAI Gym.

### Play one game with visualization

To play one game and show game screen while playing:
```
./play.sh snapshots/breakout_77.pkl
```
You can do this even without GPU, by adding `--backend cpu` to command line. During gameplay you can use following keys:
* `a` - slow down,
* `s` - speed up,
* `m` - manual control mode,
* `[` - volume down,
* `]` - volume up.

Visualization works even in text terminal!

### Record game video

To play one game and record a video:
```
./record.sh snapshots/breakout_77.pkl
```
First game frames are extracted to `videos/<game>` folder as PNG files. Then `avconv` is used to convert these into video, which is saved to `videos/<game>_<epoch_nr>.mov`.

### Plotting results

To plot results:
```
./plot.sh results/breakout.csv
```


이는 게임당 평균 보상, 단계당 게임 수(훈련, 테스트 또는 무작위), 검증 세트의 평균 Q 값 및 평균 네트워크 손실의 4가지 주요 수치를 포함하는 `results/breakout.png`를 생성합니다. `--fields` 옵션을 사용하여 플로팅 결과를 사용자 정의할 수 있습니다. 쉼표로 구분된 CSV 필드 이름을 나열합니다(첫 번째 행). 예를 들어 기본 결과는 `--fieldsaverage_reward,meanq,nr_games,meancost`를 사용하여 얻을 수 있습니다. 그림의 순서는 왼쪽에서 오른쪽, 위에서 아래입니다.

### Visualizing filters

To produce filter visualizations with guided backpropagation:

```
./nvis.sh snapshots/breakout_77.pkl
```

What the filter visualization does:

1. first it plays one game to produce a set of states (one state is 4 frames), 
2. then it finds the states which activate each filter the most,
3. finally it carries out guided backpropagation to show which parts of the screen affect the "activeness" of each filter the most. 

The result is written to file `results/<game>.html`. By default only 4 filters from each convolutional layer are visualized. To see more filters add `--visualization_filters <nr_filters>` to the command line.

NB! Because it is not very clear how to visualize the state consisting of  4 frames, I made a simplification - I'm using only the last 3 frames and putting them to different color channels. So everything that is gray hasn't changed, blue is the most recent change, then green and then red. It is easier to understand if you look at the trace of a ball - it is marked by red-green-blue.

### Nervana Cloud

To train a model with Nervana Cloud, first install and configure [Nervana Cloud](http://doc.cloud.nervanasys.com/docs/latest/ncloud.html).

Assuming the necessary dependencies are installed, run
```
ncloud train src/main.py --args "roms/breakout.bin --save_weights_prefix snapshopts/breakout --csv_file results/breakout.csv" --custom_code_url https://github.com/NervanaSystems/simple_dqn
```
This will download the repo and run the training script.

To test a model using Nervana Cloud run:
```
ncloud train src/main.py --args "roms/breakout.bin --random_steps 0 --train_steps 0 --epochs 1 --load_weights snapshops/breakout_77.pkl" --custom_code_url https://github.com/NervanaSystems/simple_dqn
```

### Profiling

프로파일링을 위한 세 가지 추가 스크립트가 있습니다.
  * `profile_train.sh` - 훈련 모드에서 Pong 게임을 1000단계 실행합니다. 미니배치 샘플링과 네트워크 훈련 코드의 병목 현상을 파악하기 위한 것입니다. 탐색 비율을 1로 설정하면 예측이 비활성화됩니다.
  * `profile_test.sh` - 테스트 모드에서 Pong 게임을 1000단계 실행합니다. 이는 예측 코드의 병목 현상을 파악하기 위한 것입니다. 탐색률을 0으로 설정하면 탐색이 비활성화됩니다.
  * `profile_random.sh` - 무작위 동작으로 탁구 게임 1000단계를 실행합니다. ALE 인터페이스의 성능을 측정하기 위한 것으로, 네트워크는 전혀 사용되지 않습니다.

### 알려진 차이점

  * Simple DQN은 Neon의 기본 RMSProp 구현을 사용하고 DeepMind는 [Alex Graves의 논문](http://arxiv.org/pdf/1308.0850v5.pdf)과 다른 공식을 사용합니다(23페이지, eq 40 참조).
  * 단순 DQN은 DeepMind 논문에서와 같이 연속된 두 프레임의 최대값 대신 건너뛴 프레임(ALE의 내장 기능) 사이에서 평균 프레임을 사용합니다.
  * Simple DQN은 Neon의 Xavier 이니셜라이저를 사용하고, DeepMind는 fan_in 매개변수 이니셜라이저를 사용합니다.
## Credits

This wouldn't have happened without inspiration and preceding work from my fellow PhD students [Kristjan Korjus](https://github.com/kristjankorjus), [Ardi Tampuu](https://github.com/RDTm), [Ilya Kuzovkin](https://github.com/kuz) and [Taivo Pungas](https://github.com/taivop) from [Computational Neuroscience lab](http://neuro.cs.ut.ee/) run by Raul Vicente in [University of Tartu](http://www.ut.ee/en), [Estonia](https://e-estonia.com/). Also I would like to thank [Nathan Sprague](https://github.com/spragunr) and other nice folks at [Deep Q-Learning list](https://groups.google.com/forum/#!forum/deep-q-learning).
