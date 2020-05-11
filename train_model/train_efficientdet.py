# refer  https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/master/train.py
import datetime
import os
import argparse
import traceback
import numpy as np
import torch
from tqdm.autonotebook import tqdm
import yaml
from tensorboardX import SummaryWriter

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.efficientdet.coco import CocoDataset, Resizer, Normalizer, Augmenter, collater
from models.efficientdet.efficient import EfficientDetBackbone
from layers.efficientdet_loss import FocalLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.efficientdet.custom_utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights
from config.efficient import efficient_config

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


project = 'coco'  # 'project file that contains parameters')
compound_coef = 0  # , help='coefficients of efficientdet')
num_workers = 12  # , help='num_workers of dataloader')
batch_size = 12  # , help='The number of images per batch among all devices')
head_only = False  # ,
#    help='whether finetunes only the regressor and the classifier, '
#        'useful in early stage convergence or small/easy dataset')
lr = 1e-4
optim = 'adamw'  # , help='select optimizer for training, '
##                                                ' very final stage then switch to \'sgd\'')
num_epochs = 500
val_interval = 1  # , help='Number of epoches between valing phases')
save_interval = 500  # , help='Number of steps between saving')
es_min_delta = 0.0  ##, help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
es_patience = 0  # help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
data_path = './datasets/'
load_weights = None
debug = False  # , help='whether visualize the predicted boxes of trainging,the output images will be in test/')

saved_path = os.path.join(efficient_config.save_weight_path , efficient_config.project_name)
log_path = os.path.join(efficient_config.save_weight_path , efficient_config.project_name)
os.makedirs(log_path, exist_ok=True)
os.makedirs(saved_path, exist_ok=True)


class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss


def train():

    if efficient_config.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)



    training_params = {'batch_size': batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': num_workers}

    val_params = {'batch_size': batch_size,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': num_workers}

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    training_set = CocoDataset(root_dir=os.path.join(data_path, efficient_config.project_name), set=efficient_config.train_set,
                        transform=transforms.Compose([Normalizer(mean=efficient_config.mean, std=efficient_config.std),
                                                      Augmenter(),
                                                      Resizer(input_sizes[compound_coef])]))
    training_generator = DataLoader(training_set, **training_params)

    val_set = CocoDataset(root_dir=os.path.join(data_path, efficient_config.project_name), set=efficient_config.val_set,
                   transform=transforms.Compose([Normalizer(mean=efficient_config.mean, std=efficient_config.std),
                                                 Resizer(input_sizes[compound_coef])]))
    val_generator = DataLoader(val_set, **val_params)

    model = EfficientDetBackbone(num_classes=len(efficient_config.obj_list), compound_coef=compound_coef,
                                 ratios=eval(efficient_config.anchors_ratios), scales=eval(efficient_config.anchors_scales))

    # load last weights
    if load_weights is not None:
        if load_weights.endswith('.pth'):
            weights_path = load_weights
        else:
            weights_path = get_last_weights(saved_path)
        try:
            last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        except:
            last_step = 0

        try:
            ret = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        last_step = 0
        print('[Info] initializing weights...')
        init_weights(model)

    # freeze backbone if train head_only
    if head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if efficient_config.num_gpus > 1 and batch_size // efficient_config.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    writer = SummaryWriter(log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, debug=debug)

    if efficient_config.num_gpus > 0:
        model = model.cuda()
        if efficient_config.num_gpus > 1:
            model = CustomDataParallel(model, efficient_config.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)

    if optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    step = max(0, last_step)
    model.train()

    num_iter_per_epoch = len(training_generator)

    try:
        for epoch in range(num_epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue

            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for iter, data in enumerate(progress_bar):
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                try:
                    imgs = data['img']
                    annot = data['annot']

                    if efficient_config.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    optimizer.zero_grad()
                    cls_loss, reg_loss = model(imgs, annot, obj_list=efficient_config.obj_list)
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                            step, epoch, num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                            reg_loss.item(), loss.item()))
                    writer.add_scalars('Loss', {'train': loss}, step)
                    writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
                    writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1

                    if step % save_interval == 0 and step > 0:
                        save_checkpoint(model, f'efficientdet-d{compound_coef}_{epoch}_{step}.pth')
                        print('checkpoint...')

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue
            scheduler.step(np.mean(epoch_loss))

            if epoch % val_interval == 0:
                model.eval()
                loss_regression_ls = []
                loss_classification_ls = []
                for iter, data in enumerate(val_generator):
                    with torch.no_grad():
                        imgs = data['img']
                        annot = data['annot']

                        if efficient_config.num_gpus == 1:
                            imgs = imgs.cuda()
                            annot = annot.cuda()

                        cls_loss, reg_loss = model(imgs, annot, obj_list=efficient_config.obj_list)
                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()

                        loss = cls_loss + reg_loss
                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        loss_classification_ls.append(cls_loss.item())
                        loss_regression_ls.append(reg_loss.item())

                cls_loss = np.mean(loss_classification_ls)
                reg_loss = np.mean(loss_regression_ls)
                loss = cls_loss + reg_loss

                print(
                    'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                        epoch, num_epochs, cls_loss, reg_loss, loss))
                writer.add_scalars('Loss', {'val': loss}, step)
                writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
                writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)

                if loss + es_min_delta < best_loss:
                    best_loss = loss
                    best_epoch = epoch

                    save_checkpoint(model, f'efficientdet-d{compound_coef}_{epoch}_{step}.pth')

                model.train()

                # Early stopping
                if epoch - best_epoch > es_patience > 0:
                    print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
                    break
    except KeyboardInterrupt:
        save_checkpoint(model, f'efficientdet-d{compound_coef}_{epoch}_{step}.pth')
        writer.close()
    writer.close()


def save_checkpoint(model, name):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), os.path.join(saved_path, name))
    else:
        torch.save(model.model.state_dict(), os.path.join(saved_path, name))


if __name__ == '__main__':
    train()
