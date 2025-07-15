# Install PyTorch, torchvision, and other dependencies
!pip install torch torchvision torchaudio
!pip install networkx einops transformers timm

# Clone YOLOv5 and install requirements
!git clone https://github.com/ultralytics/yolov5.git
%cd yolov5
%pip install -r requirements.txt
%cd ..

!git clone https://github.com/ultralytics/yolov5.git
%cd yolov5
%pip install -r requirements.txt


import torch
print(torch.cuda.is_available())  # Should return True if GPU is available
print(torch.cuda.device_count())  # Should show the count of available GPUs
print(torch.cuda.current_device())  # Should return the current device index
print(torch.cuda.get_device_name(0))  # Should return the GPU name

device = select_device('cpu')

#install dependencies

import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

device = select_device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model = DetectMultiBackend(weights='yolov5s.pt', device=device)
yolo_model.eval()


# Using YoLo v5 for detection
def detect_objects(image_path):
    img0 = cv2.imread(image_path)
    img = cv2.resize(img0, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).to(device).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    pred = yolo_model(img_tensor)[0]
    pred = non_max_suppression(pred)[0]

    detections = []
    if pred is not None:
        for *xyxy, conf, cls in pred:
            detections.append({
                "bbox": xyxy,
                "conf": conf.item(),
                "class": int(cls.item())
            })
    return Image.fromarray(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)), detections

import networkx as nx

def build_graph(detections):
    G = nx.Graph()
    for i, det in enumerate(detections):
        G.add_node(i, **det)

    for i in range(len(detections)):
        for j in range(i+1, len(detections)):
            xi1, yi1, xi2, yi2 = detections[i]["bbox"]
            xj1, yj1, xj2, yj2 = detections[j]["bbox"]
            box_i = [xi1.item(), yi1.item(), xi2.item(), yi2.item()]
            box_j = [xj1.item(), yj1.item(), xj2.item(), yj2.item()]
            dist = ((box_i[0] - box_j[0]) ** 2 + (box_i[1] - box_j[1]) ** 2) ** 0.5
            if dist < 100:  # proximity threshold
                G.add_edge(i, j, weight=1 / (dist + 1e-5))
    return G

