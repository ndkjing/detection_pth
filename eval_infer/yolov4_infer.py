# refer https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/demo.py

# import sys
# import time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from utils.yolo.utils import *
from models.yolo.yolov4 import Darknet
from config.yolo import config
# 检测图片
def detect(cfgfile, weightfile, imgfile):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    num_classes = 80
    if num_classes == 20:
        namesfile = '../config/yolo/voc.names'
    elif num_classes == 80:
        namesfile = '../config/yolo/coco.names'
    else:
        namesfile = '../config/yolo/names'

    use_cuda = 0
    if use_cuda:
        m.cuda()

    img = Image.open(imgfile).convert('RGB')
    sized = img.resize((m.width, m.height))

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.2, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    class_names = load_class_names(namesfile)
    save_image_path = os.path.join('./images_out',os.path.split(imgfile)[1].split('.')[0]+'_yolov4.jpg')
    plot_boxes(img, boxes, save_image_path, class_names)


def detect_imges(cfgfile, weightfile, imgfile_list=['data/dog.jpg', 'data/giraffe.jpg']):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    num_classes = 80
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'

    use_cuda = 0
    if use_cuda:
        m.cuda()

    imges = []
    imges_list = []
    for imgfile in imgfile_list:
        img = Image.open(imgfile).convert('RGB')
        imges_list.append(img)
        sized = img.resize((m.width, m.height))
        imges.append(np.expand_dims(np.array(sized), axis=0))

    images = np.concatenate(imges, 0)
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, images, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    class_names = load_class_names(namesfile)
    for i,(img,box) in enumerate(zip(imges_list,boxes)):
        plot_boxes(img, box, 'predictions{}.jpg'.format(i), class_names)


def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'


    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)


def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'

    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)


if __name__ == '__main__':
    cfgfile = config.yolov4_cfg_path
    weightfile = config.pre_train_weight_path
    imgfile =  '../images_test/img.png'
    detect(cfgfile, weightfile,imgfile)
    # detect_imges(cfgfile, weightfile)
    # detect_cv2(cfgfile, weightfile, imgfile)
    # detect_skimage(cfgfile, weightfile, imgfile)
