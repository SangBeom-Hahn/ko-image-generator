import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from utils import *
from flask import Flask, render_template, request, url_for
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image
from annoy import AnnoyIndex
import torch, random

clip_model = CLIPTextModel.from_pretrained("/workspace/data/models/converted", torch_dtype=torch.float16)
clip_processor = CLIPTokenizer.from_pretrained("/workspace/data/models/converted")
pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    text_encoder = clip_model,
    tokenizer = clip_processor,
    torch_dtype=torch.float16
).to("cuda")

pipeline.unet = torch.compile(pipeline.unet, mode = "reduce-overhead", fullgraph = True)
pipeline.vae = torch.compile(pipeline.vae, mode = "reduce-overhead", fullgraph = True)
pipeline.text_encoder = torch.compile(pipeline.text_encoder, mode = "reduce-overhead", fullgraph = True)

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
).to("cuda")
refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)

regenerator = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    text_encoder = clip_model,
    tokenizer = clip_processor,
    torch_dtype=torch.float16
).to("cuda")

regenerator.unet = torch.compile(regenerator.unet, mode="reduce-overhead", fullgraph=True)


app = Flask(__name__, static_folder = STATIC)

# 이런거 다 util에 
# req txt 만들기

def load_lora():
    if selected_lora != DEFAULT_LORA:
        pipeline.load_lora_weights(LORA_WEIGHTS[selected_lora])
        regenerator.load_lora_weights(LORA_WEIGHTS[selected_lora])
    else:
        pipeline.unload_lora_weights()
        regenerator.unload_lora_weights()

def generate_image(prompt, image_paths, generator = None):
    image = sd_pipeline(prompt=prompt, negative_prompt = COMMON_NEG, num_inference_steps=NUM_INFERECEN_STEPS, generator=generator).images[0]
    image = refiner(
        prompt = REFINE_PROMPT,
        negative_prompt = REFINE_NEG,
        image = image,
        num_inference_steps = NUM_INFERECEN_STEPS,
        guidance_scale = GUIDANCE_SCALE
    ).images[0]
    rand = random.randint(0, 2**32 - 1)
    image.save(f"./static/generated_image_{rand}.png")
    image_paths.append(url_for('static', filename=f'generated_image_{rand}.png'))

def get_generate_cnt(image_counts):
    return image_counts-1

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image_paths = []
        prompt, selected_lora, image_counts = extract_txt2img_request_message(request)
        load_lora()

        generate_image(prompt, image_paths, SEED)
        for _ in range(get_generate_cnt(image_counts)):
            generate_image(prompt, image_paths)
        default_prompt=prompt

        return render_template("index.html", lora_weights=LORA_WEIGHTS.keys(), images=image_paths, default_value = default_prompt)

    return render_template("index.html", lora_weights=LORA_WEIGHTS.keys())

if __name__ == "__main__":
    app.run(debug = True)