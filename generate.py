from flask import Flask, render_template, request
import os
import json
import re
from PIL import Image
import cv2
import json
import os
import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess
import io

app = Flask(__name__)
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
#model, vis_processors, _ = load_model_and_preprocess(
#    name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device
#)
model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device="cuda:1")
import time

@app.route('/generate', methods=['POST'])
def get_img():
    begin = time.time()
    f = request.files['file']
    img = f.read()
    byte_stream = io.BytesIO(img)
    raw_image = Image.open(byte_stream).convert("RGB")
    print('open img:', str(time.time() -begin))
    begin = time.time()
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    result = model.generate({"image": image, "prompt": "Find the red block and throw it in the trash"})[0]
    print('gen:', str(time.time() -begin))
    return result
    # render_template()函数是flask函数，它从模版文件夹templates中呈现给定的模板上下文。
    # print(index, image_path, img_stream)
    # return {"name":img_path, "img": img_stream}
from random import sample

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")
    # raw_image = Image.open("output/IMG_8968.MOV_0.jpg").convert('RGB')
    # print(raw_image)
