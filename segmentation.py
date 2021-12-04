import torch
from PIL import Image
import io


def get_yolov5():
    # local best.pt
    model = torch.hub.load('./yolov5', 'custom', path='./model/best.pt', source='local')  # local repo
    model.conf = 0.5
    return model


def get_image_from_bytes(binary_image, max_size=1024):
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    width, height = input_image.size
    resize_factor = min(max_size / width, max_size / height)
    resized_image = input_image.resize(
        (
            int(input_image.width * resize_factor),
            int(input_image.height * resize_factor),
        )
    )
    return resized_image
