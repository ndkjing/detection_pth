cfg = '../config/yolov3/yolov3.cfg'  # , help='*.cfg path')
names = '../config/yolo/coco.names'
model_name ="yolov3"
weights = '/Data/jing/weights/pth/yolo/pre_train/yolov3.pt'
source = '../images_test/img.png'  ## input file/folder, 0 for webcam
output = 'images_out'  # , help='output folder')  # output folder
img_size = 512  # , help='inference size (pixels)')
conf_thres = 0.3  # , help='object confidence threshold')
iou_thres = 0.6  # , help='IOU threshold for NMS')
fourcc = 'mp4v'  # , help='output video codec (verify ffmpeg support)')
half = False  # help='half precision FP16 inference')
device = "3" # ', default='', help='device id (i.e. 0 or 0,1) or cpu')
view_img = False  # ', action='store_true', help='display results')
save_txt = False  # ', action='store_true', help='save results to *.txt')
classes = ""  # ', nargs='+', type=int, help='filter by class')
agnostic_nms = False  # ', action='store_true', help='class-agnostic NMS')
augment = False  # ', action='store_true', help='augmented inference')