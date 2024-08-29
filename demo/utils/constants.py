from typing import Final

NUM_INFERECEN_STEPS: Final = 40
GUIDANCE_SCALE: Final = 7.5

REFINE_PROMPT: Final = "Realistic, Masterpiece , Photo, Super resolution , High Quality"
REFINE_NEG: Final = "(open mouth),[lowres],[smartphone camera], [amateur camera],[3d render],[sepia],((anime)),((drawn)),(paint),(teeth), deformed, bad body proportions, mutation, (ugly), disfigured,(string)"
COMMON_NEG: Final = "불완전한 신체 구조, 불명확, 잘림, 왜곡, 복사, 실수, 추가 팔, 추가 다리, 불쾌한 비율, 긴 목, 저급, 저해상도, 팔 다리 부족, 건강하지 않음, 유전적 변이, 프레임을 벗어남, 텍스트 삽입, 매력적이지 않음, 최저 품질"