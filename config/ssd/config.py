import os.path
# gets home dir cross platform home目录
HOME = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
import numpy as np
from utils.ssd.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors

dataset_type = "voc"  # Specify dataset type. Currently support voc and open_images.')
datasets_path = ["/Data/jing/VOCdevkit/VOC2012","/Data/jing/VOCdevkit/VOC2007"]  # nargs='+', help='Dataset directory path')  # 一个或者多个参数
validation_dataset = "/Data/jing/VOCdevkit/VOC2007/"  # help='Dataset directory path')
balance_data = False  # "Balance training datasets by down-sampling more frequent labels.")
net_type_lists = ["mb1_ssd", "mb1_ssd_lite", 'mb2_ssd_lite',  "mb3_ssd_lite", "vgg16_ssd"]
net_type = net_type_lists[1]  #  mb1_ssd, mb1_ssd_lite, mb2_ssd_lite  mb3_ssd_lite or vgg16_ssd.")
assert net_type in ["mb1_ssd", "mb1_ssd_lite", "mb2_ssd_lite",  "mb3_ssd_lite", "vgg16_ssd"],'net type is not in choose allow'
freeze_base_net=False#, help="Freeze base net layers.")
freeze_net=False #', action='store_true', help="Freeze all the layers except the prediction head.")
mb2_width_mult=1.0  #', default=1.0, type=float, help='Width Multiplifier for MobilenetV2')

# Params for SGD
lr=1e-3    #r', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
momentum=0.9#', default=0.9, type=float, help='Momentum value for optim')
weight_decay=5e-4  #', default=5e-4, type=float, help='Weight decay for SGD')
gamma=0.1  # ', default=0.1, type=float, help='Gamma update for SGD')
base_net_lr=None#', default=None, type=float, help='initial learning rate for base net.')
extra_layers_lr=None #'initial learning rate for the layers not in base net and prediction heads.')

# Params for loading pretrained basenet or checkpoints.
base_net=None     # help='Pretrained base model')
pretrained_ssd=None    #', help='Pre-trained base model')
resume=None  #default=None, type=str, help='Checkpoint state_dict file to resume training from')

# Scheduler
scheduler="cosine"#"Scheduler for SGD. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
milestones="80_100"# type=str, help="milestones for MultiStepLR")

# Params for Cosine Annealing
t_max=120#type=float, help='T_max value for Cosine Annealing Scheduler.')

# Train params
batch_size = 32  # type=int,help='Batch size for training')
num_epochs = 12000  # type=int, help='the number epochs')
num_workers = 4  # type=int,help='Number of workers used in dataloading')
validation_epochs = 5  # type=int, help='the number epochs')
debug_steps = 100  # type=int,help='Set the debug log output frequency.')
use_cuda = True  # type=str2bool,help='Use CUDA to train_model model')
checkpoint_folder = os.path.join(HOME, 'weights')  # help='Directory for saving checkpoint models')

device_id = 2    # 使用GPU编号若没有则指定-1或None

# 预训练模型路径
pre_train_weight_path ={'mb2_ssd_lite':"/Data/jing/weights/pth/ssd/pre_train/mb2-ssd-lite-mp-0_686.pth",
                  'mb1_ssd':"/Data/jing/weights/pth/ssd/pre_train/mobilenet-v1-ssd-mp-0_675.pth",
                  'vgg16-ssd':"/Data/jing/weights/pth/ssd/pre_train/vgg16-ssd-mp-0_7726.pth"}
# 保存模型路劲
save_weight_path="/Data/jing/weights/pth/ssd"

label_file_path = {"voc":"/home/create/jing/jing_vision/detection/pth/pth/datasets/ssd/voc-model-labels.txt",
                   "coco":"/home/create/jing/jing_vision/detection/pth/pth/datasets/ssd/coco_labels.txt",
                   "egohands":"/home/create/jing/jing_vision/detection/pth/pth/datasets/ssd/ego-models-labels.txt"}

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

image_size = 300
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

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
    'num_classes': 81,
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

class mb1_ssd:
    image_size =300
    image_mean = np.array([127, 127, 127])  # RGB layout
    image_std = 128.0
    iou_threshold = 0.45
    center_variance = 0.1
    size_variance = 0.2

    specs = [
        SSDSpec(19, 16, SSDBoxSizes(60, 105), [2, 3]),
        SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
        SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, 3]),
        SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3]),
        SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
        SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])
    ]

    priors = generate_ssd_priors(specs, image_size)

class vgg16_ssd:
    image_size = 300
    image_mean = np.array([123, 117, 104])  # RGB layout
    image_std = 1.0

    iou_threshold = 0.45
    center_variance = 0.1
    size_variance = 0.2

    specs = [
        SSDSpec(38, 8, SSDBoxSizes(30, 60), [2]),
        SSDSpec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
        SSDSpec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
        SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
        SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
        SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
    ]

    # 生成锚框
    priors = generate_ssd_priors(specs, image_size)


class squeeze_ssd:
    image_size = 300
    image_mean = np.array([127, 127, 127])  # RGB layout
    image_std = 128.0
    iou_threshold = 0.45
    center_variance = 0.1
    size_variance = 0.2

    specs = [
        SSDSpec(17, 16, SSDBoxSizes(60, 105), [2, 3]),
        SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
        SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, 3]),
        SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3]),
        SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
        SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])
    ]

    priors = generate_ssd_priors(specs, image_size)


class mb3_ssd:
    image_size = 300
    image_mean = np.array([127, 127, 127])  # RGB layout
    image_std = 128.0
    iou_threshold = 0.45
    center_variance = 0.1
    size_variance = 0.2

    specs = [
        SSDSpec(19, 16, SSDBoxSizes(60, 105), [2, 3]),
        SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
        SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, 3]),
        SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3]),
        SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
        SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])
    ]

    priors = generate_ssd_priors(specs, image_size)

net_self_config={'mb1_ssd':mb1_ssd,
                 'mb1_ssd_lite':mb1_ssd,
                 'mb2_ssd_lite':mb1_ssd,
                 'mb3_ssd_lite':mb3_ssd,
                 'vgg16_ssd':vgg16_ssd}


if __name__=="__main__":
    a='a'
    b= ['a','b','c']
    c = {
        'a':1,
        'b':2,
    }
    print(a,b,c)
    assert 'd' in c,'error keys'