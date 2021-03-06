
import shutil
from pathlib import Path
import onnx
import tensorflow as tf
from tensorflow.python.keras import backend as K
from onnx2keras import onnx_to_keras

"""
converters
"""

def pytorch2savedmodel(onnx_model_path, saved_model_dir):
    onnx_model = onnx.load(onnx_model_path)

    input_names = ['image_array']
    k_model = onnx_to_keras(onnx_model=onnx_model, input_names=input_names,
                            change_ordering=True, verbose=False)

    weights = k_model.get_weights()

    K.set_learning_phase(0)

    saved_model_dir = Path(saved_model_dir)
    if saved_model_dir.exists():
        shutil.rmtree(str(saved_model_dir))
    saved_model_dir.mkdir()

    with K.get_session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        k_model.set_weights(weights)

        tf.saved_model.simple_save(
            sess,
            str(saved_model_dir.joinpath('1')),
            inputs={'image_array': k_model.input},
            outputs=dict((output.name, tensor) for output, tensor in zip(onnx_model.graph.output, k_model.outputs))
        )


def savedmodel2tflite(saved_model_dir, tflite_model_path, quantize=False):
    saved_model_dir = str(Path(saved_model_dir).joinpath('1'))
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = converter.convert()

    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    return tflite_model



"""
tflite
"""

def get_tflite_outputs(input_array, tflite_model):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_array)
    interpreter.invoke()

    tflite_results = interpreter.get_tensor(output_details[0]['index'])
    return tflite_results

"""
image
"""
from PIL import Image
import numpy as np


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_and_preprocess_image(path, image_size=(224, 224)):
    image = Image.open(path)
    image = image.convert('RGB')
    image = image.resize(image_size)
    image_array = np.asarray(image)
    image_array = image_array.transpose((2, 0, 1))  # H x W x C --> C x H x W
    image_array = preprocess_input(image_array, MEAN, STD)
    return np.expand_dims(image_array, 0)  # C x H x W --> N x C x H x W


def preprocess_input(array, mean, std):
    array = array.astype(np.float32)
    array /= 255
    array[0, :, :] = (array[0, :, :] - mean[0]) / std[0]
    array[1, :, :] = (array[1, :, :] - mean[1]) / std[1]
    array[2, :, :] = (array[2, :, :] - mean[2]) / std[2]
    return array


"""
main
"""
import logging
from pathlib import Path
import sys
import cv2
import time
import torch
# from torchvision import transforms
# import numpy as np
from PIL import Image


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
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

    net_type = 'mb1_ssd'  # ['vgg16-ssd','mb1-ssd','mb1-ssd-lite','mb2-ssd-lite','sq-ssd-lite']
    assert net_type in config.pre_train_weight_path.keys()
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

    net.load(model_path)
    for m in net.modules():
        m.training = False

    onnx_model_path = 'model.onnx'
    dummy_input = torch.randn(1, 3, 300, 300)
    input_names = ['image_array']
    output_names = ['category']

    logger.info(f'\nExport PyTorch model in ONNX format to {onnx_model_path}.\n')
    torch.onnx.export(net, dummy_input, onnx_model_path,
                      input_names=input_names, output_names=output_names)

    saved_model_dir = 'saved_model'
    logger.info(f'\nConvert ONNX model to Keras and save as saved_model.pb.\n')
    pytorch2savedmodel(onnx_model_path, saved_model_dir)

    logger.info(f'\nConvert saved_model.pb to TFLite model.\n')
    tflite_model_path = 'model.tflite'
    tflite_model = savedmodel2tflite(saved_model_dir, tflite_model_path, quantize=False)

    logger.info(f'\nConvert saved_model.pb to TFLite quantized model.\n')
    tflite_quantized_model_path = 'model_quantized.tflite'
    tflite_quantized_model = savedmodel2tflite(saved_model_dir, tflite_quantized_model_path, quantize=True)

    logger.info("\nCompare PyTorch model's outputs and TFLite models' outputs.\n")
    num_same_outputs = 0
    image_path_list = list(Path('tools').glob('*.jpg'))
    for path in image_path_list:
        input_array = load_and_preprocess_image(str(path))
        input_tensor = torch.from_numpy(input_array)

        torch_output = model_torch(input_tensor).data.numpy().reshape(-1, )
        tflite_output = get_tflite_outputs(input_array.transpose((0, 2, 3, 1)), tflite_model).reshape(-1, )
        logger.info(f'PyTorch - first 5 items: {torch_output[:5]}')
        logger.info(f'TFLite - first 5 items: {tflite_output[:5]}')

        torch_output_index = np.argmax(torch_output)
        tflite_output_index = np.argmax(tflite_output)
        logger.info(f'PyTorch - argmax index: {torch_output_index}')
        logger.info(f'TFLite - argmax index: {tflite_output_index}\n')

        if torch_output_index == tflite_output_index:
            num_same_outputs += 1

    logger.info(f'# of matched outputs: {num_same_outputs} / {len(image_path_list)}\n')

    logger.info("\nCompare PyTorch model's outputs and TFLite quantized models' outputs.\n")
    num_same_outputs = 0
    image_path_list = list(Path('tools').glob('*.jpg'))
    for path in image_path_list:
        input_array = load_and_preprocess_image(str(path))
        input_tensor = torch.from_numpy(input_array)

        torch_output = model_torch(input_tensor).data.numpy().reshape(-1, )
        tflite_output = get_tflite_outputs(input_array.transpose((0, 2, 3, 1)), tflite_quantized_model).reshape(-1, )
        logger.info(f'PyTorch - first 5 items: {torch_output[:5]}')
        logger.info(f'TFLite - first 5 items: {tflite_output[:5]}')

        torch_output_index = np.argmax(torch_output)
        tflite_output_index = np.argmax(tflite_output)
        logger.info(f'PyTorch - argmax index: {torch_output_index}')
        logger.info(f'TFLite - argmax index: {tflite_output_index}\n')

        if torch_output_index == tflite_output_index:
            num_same_outputs += 1

    logger.info(f'# of matched outputs: {num_same_outputs} / {len(image_path_list)}\n')


if __name__ == '__main__':
    main()