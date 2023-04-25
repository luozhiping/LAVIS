import torch
from omegaconf import OmegaConf
from lavis.common.registry import registry
from lavis.models import load_preprocess
from PIL import Image
import requests
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model_cls = registry.get_model_class("blip2_opt")
model = model_cls(img_size=224,vit_precision="fp32",freeze_vit=True)
model.load_checkpoint("/root/luo6/LAVIS/lavis/output/BLIP2/Pretrain_stage2/20230402224/checkpoint_9.pth")
model.eval()
cfg = OmegaConf.load(model_cls.default_config_path("pretrain_opt2.7b"))
preprocess_cfg = cfg.preprocess
vis_processors, txt_processors = load_preprocess(preprocess_cfg)
model.to(device)
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
raw_image = raw_image.resize((224,224))
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

model.generate({"image": image})
