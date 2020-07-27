import cv2
import onnx

model = onnx.load('/home/create/jing/jing_vision/detection/pth/pth/train_model/runs/exp15/weights/best.onnx')
print(model)
model1 = cv2.dnn.readNetFromONNX('/home/create/jing/jing_vision/detection/pth/pth/train_model/runs/exp15/weights/best.onnx')
print(model1)