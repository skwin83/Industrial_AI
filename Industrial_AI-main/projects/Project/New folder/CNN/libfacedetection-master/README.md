
5,000 / 5,000
번역 결과
번역 결과
# libface 감지

이미지에서 CNN 기반 얼굴 감지를 위한 오픈 소스 라이브러리입니다. CNN 모델은 C 소스 파일에서 정적 변수로 변환되었습니다. 소스 코드는 다른 라이브러리에 의존하지 않습니다. 필요한 것은 C++ 컴파일러뿐입니다. C++ 컴파일러를 사용하면 Windows, Linux, ARM 및 모든 플랫폼에서 소스 코드를 컴파일할 수 있습니다.

SIMD 명령어는 감지 속도를 높이는 데 사용됩니다. Intel CPU 또는 ARM용 NEON을 사용하는 경우 AVX2를 활성화할 수 있습니다.

모델 파일은 `src/faceDetectcnn-data.cpp`(C++ 배열) 및 [OpenCV Zoo의 모델(ONNX)](https://github.com/opencv/opencv_zoo/tree/master/models/face_Detection_yunet)에 제공됩니다. ). ONNX 모델을 사용하여 `opencv_dnn/`에서 스크립트(C++ 및 Python)를 사용해 볼 수 있습니다. 네트워크 아키텍처는 [여기](https://netron.app/?url=https://raw.githubusercontent.com/ShiqiYu/libfaceDetection.train/master/onnx/yunet*.onnx)에서 확인하세요.

OpenCV DNN은 동적 입력 형태를 갖춘 최신 버전의 YuNet을 지원하지 않습니다. OpenCV DNN을 사용하여 최신 YuNet을 실행하려면 ONNX 모델의 입력 모양과 정확히 동일한 입력 모양이 있는지 확인하세요.

example/Detect-image.cpp 및 example/Detect-camera.cpp는 라이브러리 사용 방법을 보여줍니다.

라이브러리는 [libfaceDetection.train](https://github.com/ShiqiYu/libfaceDetection.train)에 의해 훈련되었습니다.

![예](/images/cnnresult.png "탐지 예")

## 코드 사용 방법

src/ 디렉터리의 파일을 프로젝트에 복사할 수 있습니다.
프로젝트의 다른 파일로 컴파일합니다.
소스 코드는 표준 C/C++로 작성되었습니다.
C/C++를 지원하는 모든 플랫폼에서 컴파일되어야 합니다.

몇 가지 팁:

   * FaceDetection_export.h 파일을 FaceDetectcnn.h 파일을 복사한 위치에 추가해주시고, FaceDetection_export.h 파일에 #define FACEDETECTION_EXPORT를 추가해주세요. 참조: [문제 #222](https://github.com/ShiqiYu/libfaceDetection/issues/222)
   * g++를 사용하여 소스 코드를 컴파일할 때 최적화를 활성화하려면 -O3을 추가하세요.
   * Microsoft Visual Studio를 사용하여 소스 코드를 컴파일하는 경우 '속도 최대화/-O2'를 선택하십시오.
   * OpenMP를 활성화하여 속도를 높일 수 있습니다. 그러나 가장 좋은 해결책은 다른 스레드에서 감지 기능을 호출하는 것입니다.

소스 코드를 정적 또는 동적 라이브러리로 컴파일한 다음 프로젝트에서 사용할 수도 있습니다.

## CNN-based Face Detection on Intel CPU

Using **AVX2** instructions
| Method             |Time          | FPS         |Time          | FPS         |
|--------------------|--------------|-------------|--------------|-------------|
|                    |  X64         |X64          |  X64         |X64          |
|                    |Single-thread |Single-thread|Multi-thread  |Multi-thread |
|cnn (CPU, 640x480)  |  50.02ms     |  19.99      |   6.55ms     |  152.65     |
|cnn (CPU, 320x240)  |  13.09ms     |  76.39      |   1.82ms     |  550.54     |
|cnn (CPU, 160x120)  |   3.61ms     | 277.37      |   0.57ms     | 1745.13     |
|cnn (CPU, 128x96)   |   2.11ms     | 474.60      |   0.33ms     | 2994.23     | 

Using **AVX512** instructions
| Method             |Time          | FPS         |Time          | FPS         |
|--------------------|--------------|-------------|--------------|-------------|
|                    |  X64         |X64          |  X64         |X64          |
|                    |Single-thread |Single-thread|Multi-thread  |Multi-thread |
|cnn (CPU, 640x480)  |  46.47ms     |  21.52      |   6.39ms     |  156.47     |
|cnn (CPU, 320x240)  |  12.10ms     |  82.67      |   1.67ms     |  599.31     |
|cnn (CPU, 160x120)  |   3.37ms     | 296.47      |   0.46ms     | 2155.80     |
|cnn (CPU, 128x96)   |   1.98ms     | 504.72      |   0.31ms     | 3198.63     | 

* Minimal face size ~10x10
* Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz
* Multi-thread in 16 threads and 16 processors.


## CNN-based Face Detection on ARM Linux (Raspberry Pi 4 B)

| Method             |Time          | FPS         |Time          | FPS         |
|--------------------|--------------|-------------|--------------|-------------|
|                    |Single-thread |Single-thread|Multi-thread  |Multi-thread |
|cnn (CPU, 640x480)  |  404.63ms    |  2.47       |  125.47ms    |   7.97      |
|cnn (CPU, 320x240)  |  105.73ms    |  9.46       |   32.98ms    |  30.32      |
|cnn (CPU, 160x120)  |   26.05ms    | 38.38       |    7.91ms    | 126.49      |
|cnn (CPU, 128x96)   |   15.06ms    | 66.38       |    4.50ms    | 222.28      |

* Minimal face size ~10x10
* Raspberry Pi 4 B, Broadcom BCM2835, Cortex-A72 (ARMv8) 64-bit SoC @ 1.5GHz
* Multi-thread in 4 threads and 4 processors.

## Performance on WIDER Face 
Run on default settings: scales=[1.], confidence_threshold=0.02, floating point:
```
AP_easy=0.887, AP_medium=0.871, AP_hard=0.768
```

## Author
* Shiqi Yu, <shiqi.yu@gmail.com>

## Contributors
All contributors who contribute at GitHub.com are listed [here](https://github.com/ShiqiYu/libfacedetection/graphs/contributors). 

The contributors who were not listed at GitHub.com:
* Jia Wu (吴佳)
* Dong Xu (徐栋)
* Shengyin Wu (伍圣寅)

## Acknowledgment
The work was partly supported by the Science Foundation of Shenzhen (Grant No. 20170504160426188).

## Citation

The master thesis of Mr. Wei Wu. All details of the algorithm are in the thesis. The thesis can be downloaded at [吴伟硕士毕业论文](wu-thesis-facedetect.pdf)
```
@thesis{wu2023thesisyunet,
    author      = {吴伟},
    title       = {面向边缘设备的高精度毫秒级人脸检测技术研究},
    type        = {硕士学位论文},
    institution = {南方科技大学},
    year        = {2023},
}
```

The paper for the main idea of this repository https://link.springer.com/article/10.1007/s11633-023-1423-y.

```
@article{wu2023miryunet,
	title     = {YuNet: A Tiny Millisecond-level Face Detector},
	author    = {Wu, Wei and Peng, Hanyang and Yu, Shiqi},
	journal   = {Machine Intelligence Research},
	pages     = {1--10},
	year      = {2023},
	doi       = {10.1007/s11633-023-1423-y},
	publisher = {Springer}
}
```

The survey paper on face detection to evaluate different methods. It can be open-accessed at https://ieeexplore.ieee.org/document/9580485
```
@article{feng2022face,
	author  = {Feng, Yuantao and Yu, Shiqi and Peng, Hanyang and Li, Yan-Ran and Zhang, Jianguo},
	journal = {IEEE Transactions on Biometrics, Behavior, and Identity Science}, 
	title   = {Detect Faces Efficiently: A Survey and Evaluations}, 
	year    = {2022},
	volume  = {4},
	number  = {1},
	pages   = {1-18},
	doi     = {10.1109/TBIOM.2021.3120412}
}
```

The loss used in training is EIoU, a novel extended IoU. The paper can be open-accessed at https://ieeexplore.ieee.org/document/9429909.
```
@article{peng2021eiou,
	author  = {Peng, Hanyang and Yu, Shiqi},
	journal = {IEEE Transactions on Image Processing}, 
	title   = {A Systematic IoU-Related Method: Beyond Simplified Regression for Better Localization}, 
	year    = {2021},
	volume  = {30},
	pages   = {5032-5044},
	doi     = {10.1109/TIP.2021.3077144}
}
```
