
import onnxruntime
import numpy as np
from onnxruntime.datasets import get_example


"""
验证onnx 模型
"""
sess = onnxruntime.InferenceSession("model.onnx")
input_name = sess.get_inputs()[0].name
print("Input name  :", input_name)
input_shape = sess.get_inputs()[0].shape
print("Input shape :", input_shape)
input_type = sess.get_inputs()[0].type
print("Input type  :", input_type)
outname = [output.name for output in sess.get_outputs()]
print("outputs name:",outname)



image = cv2.imread('p1.jpg')
i = cv2.resize(image,(300,300))
print(i.shape)
i = np.expand_dims(i,axis=0)
print(i.shape)














result[0][0][0]

result[0][0][0]
result[1].shape












