# ssd.py
import os.path

# gets home dir cross platform home目录
HOME = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
use_gpu_id=3
pre_train_weight_path ={"mb2-ssd-lite":"/Data/jing/weights/pth/ssd/pre_train/mb2-ssd-lite-mp-0_686.pth",
                  "mb1-ssd":"/Data/jing/weights/pth/ssd/pre_train/mobilenet-v1-ssd-mp-0_675.pth",
                  "vgg16-ssd":"/Data/jing/weights/pth/ssd/pre_train/vgg16-ssd-mp-0_7726.pth"}

save_weight_path="/Data/jing/weights/pth/ssd"

label_file_path = {"voc":"/home/create/jing/jing_vision/detection/pth/pth/datasets/ssd/voc-model-labels.txt",
                   "coco":"/home/create/jing/jing_vision/detection/pth/pth/datasets/ssd/coco_labels.txt",
                   "egohands":"/home/create/jing/jing_vision/detection/pth/pth/datasets/ssd/ego-models-labels.txt"}

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD300 CONFIGS
voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

coco = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}
