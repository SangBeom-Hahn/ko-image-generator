import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from utils import *
from flask import Flask, render_template, request, url_for
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image
from annoy import AnnoyIndex
import torch

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

def load_lora(selected_lora):
    if selected_lora != DEFAULT_LORA:
        pipeline.load_lora_weights(LORA_WEIGHTS[selected_lora])
        regenerator.load_lora_weights(LORA_WEIGHTS[selected_lora])
    else:
        pipeline.unload_lora_weights()
        regenerator.unload_lora_weights()

def generate_text_to_image_process(prompt, image_paths, generator = None):
    image = pipeline(prompt=prompt, negative_prompt = COMMON_NEG, num_inference_steps=NUM_INFERECEN_STEPS, generator=generator).images[0]
    image = refiner(
        prompt = REFINE_PROMPT,
        negative_prompt = REFINE_NEG,
        image = image,
        num_inference_steps = NUM_INFERECEN_STEPS,
        guidance_scale = GUIDANCE_SCALE
    ).images[0]

    image_save_path, render_path = get_generate_image_path()
    image.save(image_save_path)
    image_paths.append(url_for(STATIC_TEXT, filename=render_path))

def generate_image_to_image_process(prompt, image_path):
    image_paths = []
    for _ in range(IMAGE_TO_IMAGE_COUNT):
        regenerate_image = regenerator(
            prompt = prompt,
            image = load_image(image_path),
            strength = STRENGTH,
            num_inference_steps = NUM_INFERECEN_STEPS,
            guidance_scale = GUIDANCE_SCALE
        ).images[0]
        
        image = refiner(
            prompt = REFINE_PROMPT,
            negative_prompt = REFINE_NEG,
            image = regenerate_image,
            num_inference_steps = NUM_INFERECEN_STEPS,
            guidance_scale = GUIDANCE_SCALE
        ).images[0]
        
        image_save_path, render_path = get_regenerate_image_path()
        image.save(image_save_path)
        image_paths.append(url_for(STATIC_TEXT, filename=render_path))
    
    return image_paths

def get_generate_cnt(image_counts):
    return image_counts-1



app = Flask(__name__, static_folder = STATIC)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image_paths = []
        prompt, selected_lora, image_counts = extract_txt2img_request_message(request)
        load_lora(selected_lora)

        generate_text_to_image_process(prompt, image_paths, SEED)
        for _ in range(get_generate_cnt(image_counts)):
            generate_text_to_image_process(prompt, image_paths)

        return render_template("index.html", lora_weights=LORA_WEIGHTS.keys(), images=image_paths, default_value = prompt)

    return render_template("index.html", lora_weights=LORA_WEIGHTS.keys())

@app.route("/img2img", methods=["POST"])
def imageToimage():
    image_path, prompt = extract_img2img_request_message(request)
    image_paths = generate_image_to_image_process(prompt, image_path)

    return render_template("index.html", lora_weights=lora_weights.keys(), images=image_paths, default_value = prompt)

if __name__ == "__main__":
    app.run(debug = True)