from PIL import Image
import torch
from torchvision import transforms

import numpy as np
import time
import os
import cv2

from models.retinanet import retinenet
from config.retinanet import config


img_path = '../images_test/img.png'

class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]

        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]

        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


# Draws a caption above the box in an image
def draw_caption(image, box, caption):
    # print('Box is: {}'.format(box))
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

def detect_image(image_path,model_path):
    name = image_path.split('/')[-1]

    transform = transforms.Compose([
        transforms.Resize((3648, 4864)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    print(image.size())



    # 初始化模型
    model = retinenet.resnet50(80)
    # 将参数转换为cuda型
    if torch.cuda.is_available():
        model = model.cuda()
    # 加载模型
    model.load_state_dict(torch.load(model_path))

    scores_threshold = 0.05
    max_boxes_per_image = 100
    iou_threshold = 0.5
    model.training = False
    model.eval()
    unnormalize = UnNormalizer()

    with torch.no_grad():
        st = time.time()
        # print((data['image'].permute(2, 0, 1).cuda().float()).shape)
        scores, classification, transformed_anchors = model(image.cuda().float())
        print('Elapsed time: {}'.format(time.time() - st))
        idxs = np.where(scores.cpu() > 0.5)
        # img shape: 3680,4896,3
        img = np.array(255 * unnormalize(image[0, :, :, :])).copy()
        img[img < 0] = 0
        img[img > 255] = 255

        img = np.transpose(img, (1, 2, 0))
        # cv2.COLOR_BGR2RGB 将BGR格式转换成RGB格式
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            label_name = 'sapling'
            score = scores[j]
            caption = '{} {:.3f}'.format(label_name, score)
            # draw_caption(img, (x1, y1, x2, y2), label_name)
            draw_caption(img, (x1, y1, x2, y2), caption)
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
        print('Saving Result')
        cv2.imwrite('retinanet50.jpg'.format(name), img)


if __name__ == '__main__':
    detect_image(img_path,config.pre_train_weight_path)