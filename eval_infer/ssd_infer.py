import sys,os
import cv2
import time
import torch

from models.ssd.vgg_ssd import create_vgg_ssd,create_vgg_ssd_predictor
from models.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd,create_mobilenetv1_ssd_predictor
from models.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite,create_mobilenetv1_ssd_lite_predictor
from models.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite,create_mobilenetv2_ssd_lite_predictor
from models.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite,create_squeezenet_ssd_lite_predictor
from models.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite,create_mobilenetv3_ssd_lite_predictor
from utils.ssd.misc import Timer
from config.ssd import config

net_type = 'vgg16-ssd' # ['vgg16-ssd','mb1-ssd','mb1-ssd-lite','mb2-ssd-lite','sq-ssd-lite']
assert net_type in config.pre_train_weight_path.keys()
model_path = config.pre_train_weight_path[net_type]
label_path = config.label_file_path["voc"]
image_path = '../images_test/img.png'

class_names = [name.strip() for name in open(label_path).readlines()]

if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True, device_id=config.use_gpu_id)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True, device_id=config.use_gpu_id)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True, device_id=config.use_gpu_id)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True, device_id=config.use_gpu_id)
elif net_type == 'mb3-ssd-lite':
    net = create_mobilenetv3_ssd_lite(len(class_names), is_test=True, device_id=config.use_gpu_id)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True, device_id=config.use_gpu_id)
else:
    print("The net type is wrong. ")
    sys.exit(1)

net.load(model_path)

if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200,device_id=config.use_gpu_id)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200,device_id=config.use_gpu_id)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200,device_id=config.use_gpu_id)
elif net_type == 'mb2-ssd-lite':
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200,device_id=config.use_gpu_id)
elif net_type == 'mb3-ssd-lite':
    predictor = create_mobilenetv3_ssd_lite_predictor(net, candidate_size=200, device_id=config.use_gpu_id)
elif net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
else:
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)

orig_image = cv2.imread(image_path)
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
boxes, labels, probs = predictor.predict(image, 10, 0.4)

for i in range(boxes.size(0)):
    box = boxes[i, :]
    cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
    #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
    cv2.putText(orig_image, label,
                (box[0] + 20, box[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2)  # line type
path = os.path.join("./images_out/","%s_%s.jpg"%(os.path.split(image_path)[1].split(".")[0],net_type,))
cv2.imwrite(path, orig_image)
print("save image path",path)
print(f"Found {len(probs)} objects. The output image is {path}")