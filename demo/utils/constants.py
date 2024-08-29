from typing import Final
import torch, random

STATIC: Final = '/workspace/hsb/k-contents/demo/static'
ROOT: Final = "/workspace/hsb/k-contents/demo/static/"
INDEX: Final = "index.html"

NUM_INFERECEN_STEPS: Final = 40
GUIDANCE_SCALE: Final = 7.5
SEED: Final = torch.manual_seed(1147446245)
IMAGE_TO_IMAGE_COUNT: Final = 6
STRENGTH: Final = 0.75
CUDA: Final = "cuda"
PER_COUNT: Final = 1

REFINE_PROMPT: Final = "Realistic, Masterpiece , Photo, Super resolution , High Quality"
REFINE_NEG: Final = "(open mouth),[lowres],[smartphone camera], [amateur camera],[3d render],[sepia],((anime)),((drawn)),(paint),(teeth), deformed, bad body proportions, mutation, (ugly), disfigured,(string)"
COMMON_NEG: Final = "불완전한 신체 구조, 불명확, 잘림, 왜곡, 복사, 실수, 추가 팔, 추가 다리, 불쾌한 비율, 긴 목, 저급, 저해상도, 팔 다리 부족, 건강하지 않음, 유전적 변이, 프레임을 벗어남, 텍스트 삽입, 매력적이지 않음, 최저 품질"
STATIC_TEXT: Final = "static"

LORA_WEIGHTS: Final = {
    "기본 SD 1.5": None,
    "여성 한복 LoRA": "/workspace/data/Lora_weights/koreandress_female",
    "남성 한복 LoRA" : "/workspace/data/Lora_weights/hanbok_male/checkpoint-1000",
    "수묵화 LoRA": "/workspace/data/Lora_weights/ad/checkpoint-2000",
    "라바 LoRA": "/workspace/data/Lora_weights/larva",
    "구름빵 LoRA" : "/workspace/data/Lora_weights/cloudbread",
    "경복궁 LoRA" : "/workspace/data/Lora_weights/royal_gbg"
}

DEFAULT_LORA: Final = "기본 SD 1.5"

SIMILARITY: Final = 'dot'
VECTOR_SIZE: Final = 59136
DB_NAME: Final = 'hanbok.annoy'
SEARCH_SIZE: Final = 3

prompt_descriptions: Final = [
    "분홍색 가방을 들고 한복을 입은 여자가 다리 위에 서 있습니다.",
    "보라색 한복을 입은 여성이 사진을 찍기 위해 포즈를 취하고 있습니다.",
    "손에 꽃을 든 흰 한복을 입은 여자입니다.",
    "흰 한복을 입고 바닥에 앉아 있는 여자가 있습니다.",
    "분홍색과 흰색 한복을 입고 머리에 꽃을 꽂은 여성과 머리에 꽃을 꽂은 여성",
    "흰 한복을 입은 여자가 꽃다발을 들고 있습니다.",
    "흰 한복을 입은 여성이 사진을 찍기 위해 포즈를 취하고 있습니다.",
    "건물 앞에 파란색과 흰색 한복을 입은 여자가 서 있습니다.",
    "흰 한복을 입은 여성이 사진을 찍기 위해 포즈를 취하고 있습니다.",
    "사진을 찍기 위해 포즈를 취하는 긴 한복을 입은 여성의 이미지",
    "노란색 긴 한복을 입은 여성이 사진을 찍기 위해 포즈를 취하고 있습니다.",
    "긴 흰색 한복을 입은 여성이 사진을 찍기 위해 포즈를 취하고 있습니다.",
    "무대 위에 흰 한복을 입은 여자가 서 있습니다.",
    "한국 전통 한복을 입은 여성이 사진 촬영을 위해 포즈를 취하고 있습니다.",
    "긴 한복을 입은 여자가 꽃다발을 들고 있습니다.",
    "흰색과 초록색 한복을 입은 여성이 사진을 찍기 위해 포즈를 취하고 있습니다.",
    "꽃다발을 들고 있는 한복 차림의 여인이 있습니다.",
    "분홍색 한복을 입은 여성이 사진을 찍기 위해 포즈를 취하고 있습니다.",
    "머리에 베일을 쓴 웨딩 한복을 입은 여자가 있습니다.",
    "긴 흰색 한복을 입은 여성이 사진을 찍기 위해 포즈를 취하고 있습니다.",
    "파란색과 흰색 한복을 입은 여자가 들판에 서 있습니다.",
    "보라색 한복을 입은 여자가 흰 벽돌 벽에 서 있습니다.",
    "꽃다발을 들고 바닥에 앉아 있는 한복 입은 여자가 있습니다.",
    "꽃병 옆에 앉아있는 분홍색 한복을 입은 여성",
    "분홍색과 흰색 한복을 입은 여인이 꽃다발을 들고 있습니다.",
    "녹색과 흰색 한복을 입은 여성이 사진을 찍기 위해 포즈를 취하고 있습니다.",
    "파란색과 회색 한복을 입은 여성이 테이블 옆에 서 있습니다.",
    "한국 전통 한복을 입은 여성이 사진 촬영을 위해 포즈를 취하고 있습니다."
]