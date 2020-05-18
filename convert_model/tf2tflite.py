### tensorflow=2.2.0
import os
"""
tf的save model 模型转tflite模型
"""
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import tensorflow as tf
print(tf.__version__)

# Weight Quantization - Input/Output=float32
model = tf.saved_model.load('saved_model')
concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
print(model,concrete_func)
concrete_func.inputs[0].set_shape([1, 3,300, 300])  # 指定输入形状
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
# converter = tf.lite.TFLiteConverter.from_keras_model(k_model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
tflite_quant_model = converter.convert()
with open('mb2_ssd.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - human_pose_estimation_3d_0001_256x448_weight_quant.tflite")