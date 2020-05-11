dataset = "coco"  # ', help='Dataset type, must be one of csv or coco.')
coco_path = ""  # , help='Path to COCO directory')
csv_train = ""  # , help='Path to file containing training annotations (see readme)')
csv_classes = ""  # ', help='Path to file containing class list (see readme)')
csv_val = ""  # ', help='Path to file containing validation annotations (optional, see readme)')
pre_train_weight_path = "/Data/jing/weights/pth/retinanet/pre_train/retiannet_coco_resnet_50_map_0_335_state_dict.pt"
depth = 50  # ', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
epochs = 100  # ', help='Number of epochs', type=int, default=100)