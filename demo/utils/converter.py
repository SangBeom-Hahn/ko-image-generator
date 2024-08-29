from flask import request
from utils import *

def extract_txt2img_request_message(request):
    request_data = request.get_json()
    return request_data["prompt"], request_data["lora"], int(request_data["number_of_images"])

def extract_img2img_request_message(request):
    data = request.get_json()
    return data["image_path"], data["prompt"]

def extract_search_request_message(request):
    data = request.get_json()
    return data["search_prompt"]

def get_random_number():
    return random.randint(0, 2**32 - 1)

def get_generate_image_path():
    render_path = f"generated_image_{get_random_number()}.png"
    return ROOT + render_path, render_path

def get_regenerate_image_path():
    render_path = f"/regenerated_image_{get_random_number()}.png"
    return ROOT + render_path, render_path

def get_image_path(idx):
    return f"{idx}.jpg"