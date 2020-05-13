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

from config.ssd import vgg_ssd_config, squeezenet_ssd_config,config

from augment.data_preprocessing import TrainAugmentation, TestTransform

from datasets.ssd.voc_dataset import VOCDataset
from datasets.ssd.open_images import OpenImagesDataset
from datasets.ssd.egohands_dataset import EGODataset

from layers.multibox_loss import MultiboxLoss


DEVICE = torch.device("cuda:%d" % config.device_id if torch.cuda.is_available() and config.use_cuda else "cpu")
print(DEVICE)
if config.use_cuda and torch.cuda.is_available():
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
    if config.net == 'vgg16-ssd':
        create_net = create_vgg_ssd
        config = vgg_ssd_config
    elif config.net == 'mb1-ssd':
        create_net = create_mobilenetv1_ssd
    elif config.net == 'mb1-ssd-lite':
        create_net = create_mobilenetv1_ssd_lite
    elif config.net == 'sq-ssd-lite':
        create_net = create_squeezenet_ssd_lite
        config = squeezenet_ssd_config
    elif config.net == 'mb2-ssd-lite':
        create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=config.mb2_width_mult, device_id=config.device_id)
        config = mobilenetv1_ssd_config
    elif config.net == 'mb3-ssd-lite':  # mobilenet_v3还有点问题
        create_net = lambda num: create_mobilenetv3_ssd_lite(num,device_id=config.device_id)
        config = mobilenetv1_ssd_config
    else:
        print("The net type is wrong.")
        sys.exit(1)
    ##########  准备数据
    # 训练数据增强函数
    train_transform = TrainAugmentation(config.net_self_config[config.net].image_size, config.net_self_config[config.net].image_mean, config.net_self_config[config.net].image_std)
    # 训练数据标注框生成函数
    target_transform = MatchPrior(config.net_self_config[config.net].priors, config.net_self_config[config.net].center_variance,
                                  config.net_self_config[config.net].size_variance, 0.5)

    test_transform = TestTransform(config.net_self_config[config.net].image_size, config.net_self_config[config.net].image_mean, config.net_self_config[config.net].image_std)

    print("Prepare training datasets.")
    datasets = []
    for dataset_path in config.datasets_path:
        if config.dataset_type == 'voc':
            dataset = VOCDataset(dataset_path, transform=train_transform,
                                 target_transform=target_transform)
            label_file = os.path.join('../datasets', "voc-model-labels.txt")
            print(label_file)
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)  # 类别数量

        elif config.dataset_type == 'open_images':
            dataset = OpenImagesDataset(dataset_path,
                                        transform=train_transform, target_transform=target_transform,
                                        dataset_type="train_model", balance_data=config.balance_data)
            label_file = os.path.join(config.checkpoint_folder, "open-images-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            print(dataset)
            num_classes = len(dataset.class_names)

        elif config.dataset_type == 'ego':
            dataset = EGODataset(dataset_path, transform=train_transform,
                                 target_transform=target_transform)
            label_file = os.path.join(config.checkpoint_folder, "ego-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)  # 类别数量

        else:
            raise ValueError(f"Dataset type {config.dataset_type} is not supported.")
        datasets.append(dataset)
    train_dataset = ConcatDataset(datasets)
    print("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, config.batch_size,
                              num_workers=config.num_workers,
                              shuffle=True)
    print("Prepare Validation datasets.")
    if config.dataset_type == "voc":
        val_dataset = VOCDataset(config.validation_dataset, transform=test_transform,
                                 target_transform=target_transform, is_test=True)
    elif config.dataset_type == 'open_images':
        val_dataset = OpenImagesDataset(config.validation_dataset,
                                        transform=test_transform, target_transform=target_transform,
                                        dataset_type="test")
        print(val_dataset)
    elif config.dataset_type == 'ego':
        val_dataset = EGODataset(config.validation_dataset,
                                 transform=test_transform,
                                 target_transform=target_transform,
                                 is_test=True)

        print("validation dataset size: {}".format(len(val_dataset)))

    val_loader = DataLoader(val_dataset, config.batch_size,
                            num_workers=config.num_workers,
                            shuffle=False)
    print("Build network.")
    net = create_net(num_classes,device_id=config.device_id)  # 创建网络 传入类别数量
    min_loss = -10000.0
    last_epoch = -1

    base_net_lr = config.base_net_lr if config.base_net_lr is not None else config.lr
    extra_layers_lr = config.extra_layers_lr if config.extra_layers_lr is not None else config.lr
    if config.freeze_base_net:
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
    elif config.freeze_net:
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
    if config.resume:
        print(f"Resume from the model {config.resume}")
        net.load(config.resume)
    elif config.base_net:
        print(f"Init from base net {config.base_net}")
        net.init_from_base_net(config.base_net)
    elif config.pretrained_ssd:
        print(f"Init from pretrained ssd {config.pretrained_ssd}")
        net.init_from_pretrained_ssd(config.pretrained_ssd)
        print(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    net = net.to(DEVICE)
    # 损失函数
    criterion = MultiboxLoss(config.net_self_config[config.net].priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.SGD(params, lr=config.lr, momentum=config.momentum,
                                weight_decay=config.weight_decay)
    print(f"Learning rate: {config.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")

    if config.scheduler == 'multi-step':
        print("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in config.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                gamma=0.1, last_epoch=last_epoch)
    elif config.scheduler == 'cosine':
        print("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, config.t_max, last_epoch=last_epoch)
    else:
        print(f"Unsupported Scheduler: {config.scheduler}.")
        sys.exit(1)

    print("Start training from epoch ",last_epoch + 1)
    for epoch in range(last_epoch + 1, config.num_epochs):
        scheduler.step()
        train(train_loader, net, criterion, optimizer,
              device=DEVICE, debug_steps=config.debug_steps, epoch=epoch)

        if epoch % config.validation_epochs == 0 or epoch == config.num_epochs - 1:
            val_loss, val_regression_loss, val_classification_loss = eval_model(val_loader, net, criterion, DEVICE)
            print(
                "Epoch:",epoch,
                "Validation Loss:" ,val_loss,
                "Validation Regression Loss " ,val_regression_loss,
                "Validation Classification Loss:",val_classification_loss,
            )
            model_path = os.path.join(config.save_weight_path, "Epoch-%d-Loss-%f.pth"%(epoch,val_loss))
            net.save(model_path)
            print(f"Saved model" ,model_path)

