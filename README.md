# kaggle-sartorius-cell-instance-segmentation

대회 링크 : https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation/overview
블로그 포스팅 링크(1편) : https://harrykur139.tistory.com/16

## 요약

프레임워크 : MMDetection / MMSegmentation

Detection 모델로 학습 후 각각의 bounding box crop image들로 Segmentation 모델 학습

- Detection 모델 : YOLOX-x
- Segmentation 모델 : UPerNet / backbone: Swin Transformer - T

시작하기 전에 data 폴더에 데이터 셋을 넣어두면 된다. 자세한 사항은 상단 블로그 포스팅 링크 참조한다.

## 필요 패키지들 설치

나는 코랩에서 진행하여 이미 파이토치 등 환경설정은 되어있었다. 더 좋은 장비가 있다면 도커를 사용해도 좋겠다.

```
pip install openmim
mim install mmdet
pip install mmsegmentation
```

## Detection 전처리

```
cd utils
```

livecell dataset

```
python convert_to_coco_livecell.py
```

kaggle dataset

```
python convert_to_coco_kaggle.py
```

```
cd ..
```

## Detection 학습

livecell

```
python utils/detection/train.py configs/detection/yolox_x_livecell.py
```

kaggle

```
python utils/detection/train.py configs/detection/yolox_x_kaggle.py
```

## Segmentation 전처리

```
cd utils
python convert_to_mmseg.py
cd ..
```

## Segmentation 학습

```
python utils/segmentation/train.py configs/segmentation/upernet_kaggle.py
```
