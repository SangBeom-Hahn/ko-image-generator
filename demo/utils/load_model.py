from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionImg2ImgPipeline
import torch

def load_models():
    clip_model = get_clip_model()
    clip_processor = get_clip_processor()
    pipeline = get_pipeline(clip_model, clip_processor)
    refiner = get_refiner()
    regenerator = get_regenerator(clip_model, clip_processor)

    return pipeline, refiner, regenerator

def get_clip_model():
    return CLIPTextModel.from_pretrained("/workspace/data/models/converted", torch_dtype=torch.float16)

def get_clip_processor():
    return CLIPTokenizer.from_pretrained("/workspace/data/models/converted")

def get_pipeline(clip_model, clip_processor):
    pipeline = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        text_encoder = clip_model,
        tokenizer = clip_processor,
        torch_dtype=torch.float16
    ).to("cuda")
    pipeline.unet = torch.compile(pipeline.unet, mode = "reduce-overhead", fullgraph = True)
    pipeline.vae = torch.compile(pipeline.vae, mode = "reduce-overhead", fullgraph = True)
    pipeline.text_encoder = torch.compile(pipeline.text_encoder, mode = "reduce-overhead", fullgraph = True)

    return pipeline
    
def get_refiner():
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    ).to("cuda")
    refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)

    return refiner

def get_regenerator(clip_model, clip_processor):
    regenerator = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        text_encoder = clip_model,
        tokenizer = clip_processor,
        torch_dtype=torch.float16
    ).to("cuda")
    regenerator.unet = torch.compile(regenerator.unet, mode="reduce-overhead", fullgraph=True)

    return regenerator