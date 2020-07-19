import os
import yaml
yaml_path = r'D:\PythonProject\jing_vision\detection\pth\models\yolo\yolov5\yolov5s.yaml'
# a = yaml.load(yaml_path,Loader=yaml.FullLoader)
# print(a,type(a))
with open(yaml_path) as f:
    b = yaml.load(f, Loader=yaml.FullLoader)  # model dict



print(type(b),len(b['anchors'][0]))