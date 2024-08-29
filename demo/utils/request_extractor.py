from flask import request

def extract_txt2img_request_message(request):
    request_data = request.get_json()
    return request_data["prompt"], request_data["lora"], int(request_data["number_of_images"])