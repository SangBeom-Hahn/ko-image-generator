# Korean Image Generator
한국어 특화 CLIP 기반 K컨텐츠 이미지 생성 프로젝트 (롯데 이노베이트 하반기 인턴십)

## Introduction

본 프로젝트는 한국어 텍스트를 이해하고, 해당 텍스트에 맞게 이미지를 생성할 수 있는 Stable Diffusion 모델을 생성한다. 기존 영어 이미지 생성 모델들에 한국어를 사용하기 위해 번역 등 추가 로직을 적용하여 발생하는 Latency를 해소하고 속도와 퀄리티를 모두 만족하는 한국어 기반 이미지 생성 서비스를 제공한다.

## ✨ Demo

### Text to Image
<p align ="center">
  <img src = "https://github.com/user-attachments/assets/ab1bf4ba-5b94-4439-aec0-d31591079c7d">
</p>

<p align ="center">
  <img src = "https://github.com/user-attachments/assets/b62c9331-7151-4837-b534-7dc25d155f1d">
</p>

### Image to Image
<p align ="center">
  <img src = "https://github.com/user-attachments/assets/0efd9920-0920-4dcd-a1d5-d5cbc95ddbf1">
</p>

<p align ="center">
  <img src = "https://github.com/user-attachments/assets/de3e38e1-33cf-4a41-b3ed-8ad7224e62b6">
</p>


### Vector Search
<p align ="center">
  <img src = "https://github.com/user-attachments/assets/c48a7e5f-2568-4c4a-8e94-a3620f0e06c9">
</p>

## Dataset

|Data|데이터 수|세부사항|
|:-:|:-:|:-:|
|1|130만|Common 한글-영어 번역 말뭉치 데이터|
|2|130만|기술 과학 한글-영어 번역 말뭉치 데이터|
|3|130만|사회 과학 한글-영어 번역 말뭉치 데이터|

학습에는 한글-영문 번역 말뭉치 데이터 셋을 활용, 카테고리 별로 약 100만개로 구성된다.

출처 : https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=126

## Model

![project_pipeline](https://github.com/SangBeom-Hahn/Sketch2Fashion/blob/main/assests/model.png)

- Image Generate

전체적인 파이프라인은 2 stage로 이루어져 있다. 1단계는 Text To Image 이미지 생성기를 통해 이미지를 생성한다. 2단계에서는 SDXL Image2Image Refiner로 1단계에서 나온 결과물을 정제하여 이미지 퀄리티를 증가시킨다. (추론 시간 : 약 4초 소요, VRAM 소모량 : 2.6GB)

- Vector Search

입력 프롬프트를 CLIP Text Encoder에 입력하여 임베딩한다. 임베딩 결과를 검색 조건으로 하여 Annoy Vector DB에서 유사한 프름포트를 검색하여 이미지와 함께 제공한다. (추론 시간 : 약 1초 소요)

## Project Structure

```
Korean Image Generator
├── demo
    ├── static
    ├── templates
    ├── utils
    ├── requirements.txt
    ├── vector_search.py
    └── app.py
└── ko_clip
    ├── config
    ├── dataset
    ├── main.py
    └── train.py
```

- demo : 어플리케이션 루트 폴더
    - static : 웹페이지 구성에 필요한 리소스를 모아놓은 폴더
    - templates : 웹페이지 템플릿 폴더
    - utils : 라우팅 유틸리티 구현 폴더
    - vector_search.py : 벡터 유사도 검색 코드
    - app.py : 라우팅 API 코드
 
- ko_clip : 한국어 CLIP 학습 폴더
    - config : 학습 파라미터 설정 폴더
    - dataset : 데이터 전처리 폴더
    - train.py : 모델 train 코드

## Requirements
```
pytorch==2.x
transformers==4.x
annoy==1.17.3
diffusers==0.30.0
xformers=0.0.26
Flask==3.0.3
```

## reference
- [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
- [Bingsu/clip-vit-large-patch14-ko](https://huggingface.co/Bingsu/clip-vit-large-patch14-ko)
- [stabilityai/stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2)
- [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
  
## Author

👤 **SangBoem-Hahn**
- Blog : [한국어 특화 CLIP 기반 k컨텐츠 이미지 생성](https://hsb422.tistory.com/entry/%E3%85%81-2)
