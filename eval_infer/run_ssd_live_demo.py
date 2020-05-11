import sys
import cv2
import time
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from ssd.vgg_ssd import create_vgg_ssd,create_vgg_ssd_predictor
from ssd.mobilenetv1_ssd import create_mobilenetv1_ssd,create_mobilenetv1_ssd_predictor
from ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite,create_mobilenetv1_ssd_lite_predictor
from models.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite,create_mobilenetv2_ssd_lite_predictor
from ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite,create_squeezenet_ssd_lite_predictor
from utils.ssd.misc import Timer

net_type = 'mb2-ssd-lite'
model_path = './weights/mb2-ssd-lite-Epoch-635-Loss-1.168032169342041.pth'
# model_path = './weights/mb1-ssd-lite-Epoch-0-Loss-5.086438020070394.pth'
label_path = './weights/ego-models-labels.txt'
image_path = './egohands/JPEGImages/JENGA_COURTYARD_B_H_frame_0010.jpg'
cap = cv2.VideoCapture(0)   # capture from camera
# cap.set(3, 1920)
# cap.set(4, 1080)
class_names = [name.strip() for name in open(label_path).readlines()]

if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True, device=torch.device('cpu'))
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
else:
    print("The net type is wrong. ")
    sys.exit(1)

net.load(model_path)
net = net.eval()
with torch.no_grad():

    if net_type == 'vgg16-ssd':
        predictor = create_vgg_ssd_predictor(net, candidate_size=200)
    elif net_type == 'mb1-ssd':
        predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
    elif net_type == 'mb1-ssd-lite':
        predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
    elif net_type == 'mb2-ssd-lite':
        predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
    elif net_type == 'sq-ssd-lite':
        predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
    else:
        predictor = create_vgg_ssd_predictor(net, candidate_size=200)

    model_classify = torch.load('./weights/sq_torch_1_0_color.pth',map_location='cpu')
    model_classify = model_classify.eval()



    trans=transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    timer = Timer()
    while True:
        ret, orig_image = cap.read()
        if orig_image is None:
            continue
        h,w= orig_image.shape[0],orig_image.shape[1]
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

        timer.start()
        boxes, labels, probs = predictor.predict(image, 10, 0.4)
        interval = timer.end()
        print(type(boxes))
        boxes,labels,probs = np.asarray(boxes),np.asarray(labels),np.asarray(probs)
        # 输出检测信息
        print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, len(labels)))
        for i in range(len(boxes)):
            box = boxes[i, :]
            label = str(class_names[labels[i]])+ str(probs[i])

            # 以检测框为中心扩大输入的图片并判断不超出图片边界
            if int(box[1])-200<=0:
                y_min_crop = 0
            else:
                y_min_crop=int(box[1])-200
            if int(box[3])+50 >= h:
                y_max_crop = h -1
            else:
                y_max_crop=int(box[3])+50
            if int(box[0])-50<=0:
                x_min_crop =0
            else:
                x_min_crop=int(box[0])-50
            if int(box[2])+50>=w:
                x_max_crop = w-1
            else:
                x_max_crop = int(box[2])+50
            calss_start_time= time.time()
            crop_image = image[y_min_crop:y_max_crop,x_min_crop:x_max_crop,:]
            class_image = Image.fromarray(np.uint8(crop_image))
            class_image = trans(class_image)
            class_image = class_image.unsqueeze(0)
            pre = model_classify(class_image.float())

            pre_index = np.argmax(pre.cpu().detach().numpy())
            calss_end_time = time.time()
            print(pre, pre_index)
            print('class time',calss_end_time-calss_start_time)
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

            cv2.putText(orig_image, label,
                        (int(box[0])+20, int(box[1])+40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type
        # cv2.imshow('annotated', orig_image)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    cap.release()
    cv2.destroyAllWindows()


