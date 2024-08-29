import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from flask import Flask, render_template, request, url_for,
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