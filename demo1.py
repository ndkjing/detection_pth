import os
import yaml
yaml_path = r'D:\PythonProject\jing_vision\detection\pth\models\yolo\yolov5\yolov5s.yaml'
# a = yaml.load(yaml_path,Loader=yaml.FullLoader)
# print(a,type(a))
# with open(yaml_path) as f:
#     b = yaml.load(f, Loader=yaml.FullLoader)  # model dict
#
# print(type(b),len(b['anchors'][0]))


import onnx

import time
onnx_path = '/home/create/jing/jing_vision/detection/pth/pth/train_model/runs/exp15/weights/best.onnx'

onnx_model =onnx.load(onnx_path)
print(len(onnx_model.graph.node))
for out in onnx_model.graph.node:
    d = out
    print(d.attribute.pop())
    print(type(d.attribute.pop()))
    print(dir(d.attribute))
    print(d.name)

    # print(d.output.type)
    # d = out.type
    # print(d)
