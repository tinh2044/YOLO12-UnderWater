import torch
# torch.backends.cuda.matmul.allow_tf32 = True  
# torch.backends.cudnn.benchmark = True 
# torch.use_deterministic_algorithms(True, warn_only=False)

from yolo12 import YOLO
from pprint import pprint

model = YOLO("yolo12/cfg/models/12/yolo12.yaml")
model.export(format="onnx",imgsz=640)


weights = torch.load("./outputs/Brackish/weights/best.pt")

pprint(weights["model"])

torch.save(weights["model"].state_dict(), "./brackish_weights.pt")


# x = torch.randn(1, 3, 640, 640)

# print(model(x))


# print(model)