### 허브 생육 상태 분류 모델

원천 데이터: [Pudina Leaf Dataset: Freshness Analysis](https://data.mendeley.com/datasets/nvbpydc3fs/1)

- Dried, Spoiled, Fresh, 세가지 상태로 분류되어 있음
- stratify를 사용하여 클래스 분포 비율을 맞추어 train/validation/test 데이터셋으로 나눔

![dataset_split](https://github.com/user-attachments/assets/c05af06b-288b-4cd6-9759-29225f7d177b)

모델: Vision Transformer pretrained model - 'vit_base_patch16_224'

<img width="640" alt="다운로드" src="https://github.com/user-attachments/assets/9818e53b-7a1f-4e1b-aa4c-79f6660285f8">

학습결과: 분류 정확도 99%

