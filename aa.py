import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device
)
raw_image = Image.open("/export/home/.cache/lavis/block/images/IMG_8886.MOV_142.jpg").convert('RGB')
#img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png'
#raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
print(model.generate({"image": image}))
