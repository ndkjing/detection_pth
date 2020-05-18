convert_onnx = True
convert_keras = True
convert_tflite = True

#
if convert_onnx:
    import sys, os
    import cv2
    import time
    import torch

    from models.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
    from models.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
    from models.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
    from models.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
    from models.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
    from models.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite, create_mobilenetv3_ssd_lite_predictor
    from utils.ssd.misc import Timer
    from config.ssd import config

    net_type = 'mb2_ssd_lite'  # ['vgg16_ssd','mb1_ssd','mb1_ssd_lite','mb2_ssd_lite','sq_ssd_lite']
    print(type(net_type), type(list(config.pre_train_weight_path.keys())))
    print(net_type, list(config.pre_train_weight_path.keys()))

    assert net_type in config.pre_train_weight_path, 'wrong net type'
    # assert net_type in [i.strip('\'') for i in list(config.pre_train_weight_path.keys())],'wrong net type'
    model_path = config.pre_train_weight_path[net_type]
    label_path = config.label_file_path["voc"]
    image_path = '../images_test/img.png'

    class_names = [name.strip() for name in open(label_path).readlines()]

    if net_type == 'vgg16_ssd':
        net = create_vgg_ssd(len(class_names), is_test=True, device_id=config.device_id)
    elif net_type == 'mb1_ssd':
        net = create_mobilenetv1_ssd(len(class_names), is_test=True, device_id=config.device_id)
    elif net_type == 'mb1_ssd_lite':
        net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True, device_id=config.device_id)
    elif net_type == 'mb2_ssd_lite':
        net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True, device_id=config.device_id)
    elif net_type == 'mb3_ssd_lite':
        net = create_mobilenetv3_ssd_lite(len(class_names), is_test=True, device_id=config.device_id)
    elif net_type == 'sq_ssd_lite':
        net = create_squeezenet_ssd_lite(len(class_names), is_test=True, device_id=config.device_id)
    else:
        print("The net type is wrong. ")
        sys.exit(1)

    print('load weights')
    net.load(model_path)

    # print(net)
    print(dir(net))
    net = net.cpu()
    dummy_input = torch.randn(1, 3, 300, 300, device='cpu')
    # 转为onnx模型
    input_names = ['input_image']
    # output_names = ['output_1', 'output_2']
    net = net.cpu()
    torch.onnx.export(net,
                      dummy_input,
                      "temp.onnx",
                      verbose=True,
                      input_names=input_names,
                      # output_names=output_names,
                      )

# 转为keras模型
if convert_keras:
    import onnx
    from onnx2keras import onnx_to_keras
    import tensorflow as tf
    import shutil

    onnx_model = onnx.load("squeeze.onnx")
    # print(onnx_model)
    k_model = onnx_to_keras(onnx_model=onnx_model, input_names=['input_1'], change_ordering=True)

    shutil.rmtree('saved_model', ignore_errors=True)
    tf.saved_model.save(k_model, 'keras_saved_model')
    print('success')







# 转为tflitem模型

### tensorflow=2.2.0
if convert_tflite:
    import tensorflow as tf

    # Weight Quantization - Input/Output=float32
    converter = tf.lite.TFLiteConverter.from_saved_model('keras_saved_model')
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_quant_model = converter.convert()
    with open('human_pose_estimation_3d_0001_256x448_weight_quant.tflite', 'wb') as w:
        w.write(tflite_quant_model)
    print("Weight Quantization complete! - human_pose_estimation_3d_0001_256x448_weight_quant.tflite")

