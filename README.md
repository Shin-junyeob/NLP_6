[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/HS6nBbT4)
# 프로젝트 이름
### Dialogue Summarization | 일상 대화 요약

<br>

## 팀 구성원

| ![김시진](https://avatars.githubusercontent.com/u/46598332?v=4) | ![신준엽](https://avatars.githubusercontent.com/u/180160571?v=4) | ![이가은](https://avatars.githubusercontent.com/u/217889143?v=4) | ![이건희](https://avatars.githubusercontent.com/u/213379929?v=4) | ![이찬](https://avatars.githubusercontent.com/u/100181857?v=4) | ![송인섭](https://avatars.githubusercontent.com/u/22423127?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [김시진](https://github.com/kimsijin33)             |            [신준엽](https://github.com/Shin-junyeob)             |            [이가은](https://github.com/kkaeunii)             |            [이건희](https://github.com/GH-Lee33)             |            [이찬](https://github.com/SKKULEE)             |            [송인섭](https://github.com/SongInseob)             |
|                            팀장, 담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |

<br>

## 1. 개발 환경 및 기술 스택
- 주 언어 : python
- 버전 및 이슈관리 : github
- 협업 툴 : github, slack

<br>

## 2. 프로젝트 구조
```
NLP6/
├─ config/
│   └─ config.yaml
│
├─ data/                # Kernel Academy 경진대회 데이터 사용
│
├─ EDA/
│   └─ main.py          # EDA
│
├─ src/
│   ├─ config.py        # Generate / Read config.yaml 
│   ├─ utils.py         # Settings
│   ├─ preprocess.py    # Preprocessing
│   ├─ model.py         # Model Architecture
│   ├─ train.py         # Training
│   └─ inference.py     # Inference
│
├─ outputs/
│   ├─ checkpoints/
│   └─ predictions/
│       └─ output.csv
│
├─ .env.sample
└─ requirements.txt

```

### How to Run 🚀
```bash
user@user:~/NLP_6# python EDA/main.py train > EDA/train_eda.csv      # train.csv 파일 EDA 및 저장
user@user:~/NLP_6# python EDA/main.py dev > EDA/dev_eda.csv          # dev.csv 파일 EDA 및 저장


user@user:~/NLP_6# python src/inference.py                           # 모델 학습 및 추론
```


<br>

## 3. 구현 기능
### 기능1
- _작품에 대한 주요 기능을 작성해주세요_
### 기능2
- _작품에 대한 주요 기능을 작성해주세요_
### 기능3
- _작품에 대한 주요 기능을 작성해주세요_

<br>

## 4. 작품 아키텍처(필수X)
- #### _아래 이미지는 예시입니다_
![이미지 설명](https://www.cadgraphics.co.kr/UPLOAD/editor/2024/07/04//2024726410gH04SyxMo3_editor_image.png)

<br>

## 5. 트러블 슈팅
### 1. OOO 에러 발견

#### 설명
- _프로젝트 진행 중 발생한 트러블에 대해 작성해주세요_

#### 해결
- _프로젝트 진행 중 발생한 트러블 해결방법 대해 작성해주세요_

<br>

## 6. 프로젝트 회고
### 박패캠
- _프로젝트 회고를 작성해주세요_

<br>

## 7. 참고자료
- _참고자료를 첨부해주세요_
