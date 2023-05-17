import json

import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device
)
#raw_image = Image.open("/export/home/.cache/lavis/block/images/IMG_8886.MOV_142.jpg").convert('RGB')
#img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png'
#raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
#image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
#print(model.generate({"image": image}))
correct = 0
error = 0
datas = json.loads(open('/export/home/.cache/lavis/dajiang_command/annotations/test.json', 'r').read())
for data in datas:
    #print(data['image_id'])
    raw_image = Image.open("/export/home/.cache/lavis/dajiang_command/images/" + data['image']).convert('RGB')
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    #print(data['caption'])
    result = model.generate({"image": image, "prompt": "Find the red block and throw it in the trash"})[0]
    print(data['image_id'], data['answer'], result)
    if data['answer'].lower().replace(".", "") == result.lower():
        correct += 1
    else:
        error += 1
print(error/len(datas))
