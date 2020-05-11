# refer:   https://github.com/qfgaohao/pytorch-ssd

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))))
import os
import sys
import itertools

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from utils.ssd.misc import Timer, freeze_net_layers, store_labels
from models.ssd.ssd import MatchPrior
from models.ssd.vgg_ssd import create_vgg_ssd
from models.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from models.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from models.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from models.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from models.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite
from config.ssd import vgg_ssd_config, squeezenet_ssd_config, mobilenetv1_ssd_config, config
from augment.data_preprocessing import TrainAugmentation, TestTransform

from datasets.ssd.voc_dataset import VOCDataset
from datasets.ssd.open_images import OpenImagesDataset
from datasets.ssd.egohands_dataset import EGODataset

from layers.multibox_loss import MultiboxLoss

dataset_type = "voc"  # Specify dataset type. Currently support voc and open_images.')
datasets_path = ["/Data/jing/VOCdevkit/VOC2012","/Data/jing/VOCdevkit/VOC2007"]  # nargs='+', help='Dataset directory path')  # 一个或者多个参数
validation_dataset = "/Data/jing/VOCdevkit/VOC2007/"  # help='Dataset directory path')
balance_data = False  # "Balance training datasets by down-sampling more frequent labels.")
net = "mb3-ssd-lite"  #  mb1-ssd, mb1-lite-ssd, mb2-ssd-lite  mb3-ssd-lite or vgg16-ssd.")
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
num_epochs = 120  # type=int, help='the number epochs')
num_workers = 4  # type=int,help='Number of workers used in dataloading')
validation_epochs = 5  # type=int, help='the number epochs')
debug_steps = 100  # type=int,help='Set the debug log output frequency.')
use_cuda = True  # type=str2bool,help='Use CUDA to train_model model')
checkpoint_folder = os.path.join(config.HOME, 'weights')  # help='Directory for saving checkpoint models')

device_id = 3
DEVICE = torch.device("cuda:%d" % device_id if torch.cuda.is_available() and use_cuda else "cpu")
print(DEVICE)
if use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print("Use Cuda.")


# 训练模型
def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            print(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0


def eval_model(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == '__main__':
    timer = Timer()

    # 初始化网络
    if net == 'vgg16-ssd':
        create_net = create_vgg_ssd
        config = vgg_ssd_config
    elif net == 'mb1-ssd':
        create_net = create_mobilenetv1_ssd
        config = mobilenetv1_ssd_config
    elif net == 'mb1-ssd-lite':
        create_net = create_mobilenetv1_ssd_lite
        config = mobilenetv1_ssd_config
    elif net == 'sq-ssd-lite':
        create_net = create_squeezenet_ssd_lite
        config = squeezenet_ssd_config
    elif net == 'mb2-ssd-lite':
        create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=mb2_width_mult, device_id=device_id)
        config = mobilenetv1_ssd_config
    elif net == 'mb3-ssd-lite':  # mobilenet_v3还有点问题
        create_net = lambda num: create_mobilenetv3_ssd_lite(num)
        config = mobilenetv1_ssd_config
    else:
        print("The net type is wrong.")
        sys.exit(1)
    ##########  准备数据
    # 训练数据增强函数
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    # 训练数据标注框生成函数
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    print("Prepare training datasets.")
    datasets = []
    for dataset_path in datasets_path:
        if dataset_type == 'voc':
            dataset = VOCDataset(dataset_path, transform=train_transform,
                                 target_transform=target_transform)
            label_file = os.path.join('../datasets', "voc-model-labels.txt")
            print(label_file)
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)  # 类别数量

        elif dataset_type == 'open_images':
            dataset = OpenImagesDataset(dataset_path,
                                        transform=train_transform, target_transform=target_transform,
                                        dataset_type="train_model", balance_data=balance_data)
            label_file = os.path.join(checkpoint_folder, "open-images-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            print(dataset)
            num_classes = len(dataset.class_names)

        elif dataset_type == 'ego':
            dataset = EGODataset(dataset_path, transform=train_transform,
                                 target_transform=target_transform)
            label_file = os.path.join(checkpoint_folder, "ego-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)  # 类别数量

        else:
            raise ValueError(f"Dataset type {dataset_type} is not supported.")
        datasets.append(dataset)
    train_dataset = ConcatDataset(datasets)
    print("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, batch_size,
                              num_workers=num_workers,
                              shuffle=True)
    print("Prepare Validation datasets.")
    if dataset_type == "voc":
        val_dataset = VOCDataset(validation_dataset, transform=test_transform,
                                 target_transform=target_transform, is_test=True)
    elif dataset_type == 'open_images':
        val_dataset = OpenImagesDataset(validation_dataset,
                                        transform=test_transform, target_transform=target_transform,
                                        dataset_type="test")
        print(val_dataset)
    elif dataset_type == 'ego':
        val_dataset = EGODataset(validation_dataset,
                                 transform=test_transform,
                                 target_transform=target_transform,
                                 is_test=True)

        print("validation dataset size: {}".format(len(val_dataset)))

    val_loader = DataLoader(val_dataset, batch_size,
                            num_workers=num_workers,
                            shuffle=False)
    print("Build network.")
    net = create_net(num_classes)  # 创建网络 传入类别数量
    min_loss = -10000.0
    last_epoch = -1

    base_net_lr = base_net_lr if base_net_lr is not None else lr
    extra_layers_lr = extra_layers_lr if extra_layers_lr is not None else lr
    if freeze_base_net:
        print("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        print("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]

    timer.start("Load Model")
    if resume:
        print(f"Resume from the model {resume}")
        net.load(resume)
    elif base_net:
        print(f"Init from base net {base_net}")
        net.init_from_base_net(base_net)
    elif pretrained_ssd:
        print(f"Init from pretrained ssd {pretrained_ssd}")
        net.init_from_pretrained_ssd(pretrained_ssd)
        print(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    net = net.to(DEVICE)
    # 损失函数
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum,
                                weight_decay=weight_decay)
    print(f"Learning rate: {lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")

    if scheduler == 'multi-step':
        print("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                gamma=0.1, last_epoch=last_epoch)
    elif scheduler == 'cosine':
        print("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, t_max, last_epoch=last_epoch)
    else:
        print(f"Unsupported Scheduler: {scheduler}.")
        sys.exit(1)

    print("Start training from epoch ",last_epoch + 1)
    for epoch in range(last_epoch + 1, num_epochs):
        scheduler.step()
        train(train_loader, net, criterion, optimizer,
              device=DEVICE, debug_steps=debug_steps, epoch=epoch)

        if epoch % validation_epochs == 0 or epoch == num_epochs - 1:
            val_loss, val_regression_loss, val_classification_loss = eval_model(val_loader, net, criterion, DEVICE)
            print(
                "Epoch:",epoch,
                "Validation Loss:" ,val_loss,
                "Validation Regression Loss " ,val_regression_loss,
                "Validation Classification Loss:",val_classification_loss,
            )
            model_path = os.path.join(checkpoint_folder, "Epoch-%d-Loss-%f.pth"%(epoch,val_loss))
            net.save(model_path)
            print(f"Saved model" ,model_path)

