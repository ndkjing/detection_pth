import os
project_name = "coco"  # also the folder name of the dataset that under data_path folder
train_set = "train2017"
val_set = "val2017"
num_gpus = "3"

# class Params:
#     def __init__(self, project_file):
#         self.params = yaml.safe_load(open(project_file).read())
#
#     def __getattr__(self, item):
#         return self.params.get(item, None)


project = 'coco'  # 'project file that contains parameters')
compound_coef = 1  # , help='coefficients of efficientdet')
num_workers = 12  # , help='num_workers of dataloader')
batch_size = 4  # , help='The number of images per batch among all devices')
head_only = False  # ,
#    help='whether finetunes only the regressor and the classifier, '
#        'useful in early stage convergence or small/easy dataset')
lr = 1e-4
# optim = 'adamw'  # , help='select optimizer for training, '
optim = 'adam'  # , help='select optimizer for training, '
##                                                ' very final stage then switch to \'sgd\'')
num_epochs = 500
val_interval = 1  # , help='Number of epoches between valing phases')
save_interval = 500  # , help='Number of steps between saving')
es_min_delta = 0.0  ##, help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
es_patience = 0  # help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
data_path = '/Data/jing'

# 预训练权重路径
pre_train_weight_path = {0: "/Data/jing/weights/detection/pth/efficientdet/pre_train/efficientdet-d0.pth",
                         1: "/Data/jing/weights/detection/pth/efficientdet/pre_train/efficientdet-d1.pth",
                         2: "/Data/jing/weights/detection/pth/efficientdet/pre_train/efficientdet-d2.pth",
                         3: "/Data/jing/weights/detection/pth/efficientdet/pre_train/efficientdet-d3.pth",
                         4: "/Data/jing/weights/detection/pth/efficientdet/pre_train/efficientdet-d4.pth",
                         5: "/Data/jing/weights/detection/pth/efficientdet/pre_train/efficientdet-d5.pth",
                         6: "/Data/jing/weights/detection/pth/efficientdet/pre_train/efficientdet-d6.pth",
                         7: "/Data/jing/weights/detection/pth/efficientdet/pre_train/efficientdet-d7.pth"
                         }
# 训练权重保存路径
save_weight_path = "/Data/jing/weights/pth/efficientdet"

load_weights =pre_train_weight_path[compound_coef]
debug = False  # , help='whether visualize the predicted boxes of trainging,the output images will be in test/')

saved_path = os.path.join(save_weight_path, project_name)
log_path = os.path.join(save_weight_path, project_name)


os.makedirs(log_path, exist_ok=True)
os.makedirs(saved_path, exist_ok=True)
# mean and std in RGB order, actually this part should remain unchanged as long as your dataset is similar to coco.
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# this is coco anchors, change it if necessary
anchors_scales = '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'
anchors_ratios = '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'

COCO_CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
                "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                "teddy bear", "hair drier", "toothbrush"]

obj_list = [ "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
            "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
            "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
            "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"]

colors = [(39, 129, 113), (164, 80, 133), (83, 122, 114), (99, 81, 172), (95, 56, 104), (37, 84, 86), (14, 89, 122),
          (80, 7, 65), (10, 102, 25), (90, 185, 109), (106, 110, 132), (169, 158, 85), (188, 185, 26), (103, 1, 17),
          (82, 144, 81), (92, 7, 184), (49, 81, 155), (179, 177, 69), (93, 187, 158), (13, 39, 73), (12, 50, 60),
          (16, 179, 33), (112, 69, 165), (15, 139, 63), (33, 191, 159), (182, 173, 32), (34, 113, 133), (90, 135, 34),
          (53, 34, 86), (141, 35, 190), (6, 171, 8), (118, 76, 112), (89, 60, 55), (15, 54, 88), (112, 75, 181),
          (42, 147, 38), (138, 52, 63), (128, 65, 149), (106, 103, 24), (168, 33, 45), (28, 136, 135), (86, 91, 108),
          (52, 11, 76), (142, 6, 189), (57, 81, 168), (55, 19, 148), (182, 101, 89), (44, 65, 179), (1, 33, 26),
          (122, 164, 26), (70, 63, 134), (137, 106, 82), (120, 118, 52), (129, 74, 42), (182, 147, 112), (22, 157, 50),
          (56, 50, 20), (2, 22, 177), (156, 100, 106), (21, 35, 42), (13, 8, 121), (142, 92, 28), (45, 118, 33),
          (105, 118, 30), (7, 185, 124), (46, 34, 146), (105, 184, 169), (22, 18, 5), (147, 71, 73), (181, 64, 91),
          (31, 39, 184), (164, 179, 33), (96, 50, 18), (95, 15, 106), (113, 68, 54), (136, 116, 112), (119, 139, 130),
          (31, 139, 34), (66, 6, 127), (62, 39, 2), (49, 99, 180), (49, 119, 155), (153, 50, 183), (125, 38, 3),
          (129, 87, 143), (49, 87, 40), (128, 62, 120), (73, 85, 148), (28, 144, 118), (29, 9, 24), (175, 45, 108),
          (81, 175, 64), (178, 19, 157), (74, 188, 190), (18, 114, 2), (62, 128, 96), (21, 3, 150), (0, 6, 95),
          (2, 20, 184), (122, 37, 185)]
