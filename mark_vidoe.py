import cv2

import json
import os
import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess
filename = "./WYWM0746.MP4"

cap = cv2.VideoCapture(filename)
i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    #cv2.imshow('capture', frame)
    cv2.imwrite("./images/%s_%s.jpg" % (filename, i), frame)
    # img = cv2.imread("./output/%s_%s.jpg" % (filename, i))
    # cv2.putText(img, "test test", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1, cv2.LINE_AA)
    # cv2.imshow("aa", img)
    # cv2.waitKey()
    i = i + 1
    print(i)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device
)
correct = 0
error = 0
results = {}

for root, dirs, files in os.walk("./images/", topdown=False):
    for name in files:
        filename = os.path.join(root, name)
    #print(data['image_id'])
        print(filename)
        raw_image = Image.open(filename).convert('RGB')
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        #print(data['caption'])
        result = model.generate({"image": image})[0]
        print(result)
        results[filename] = result

print(results)
exit()
data_path = "./"
fps = 30  # 视频帧率
size = (1080, 1920)  # 需要转为视频的图片的尺寸

for i in range(len(results)):
    img = cv2.imread("./output/%s_%s.jpg" % (filename, i))
    cv2.putText(img, results[i], (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1, cv2.LINE_AA)
    # cv2.imshow("aa", img)
    cv2.imwrite("./output/%s_%s.jpg" % (filename, i))
    # cv2.waitKey()
    print(i, results[i])

