import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))),"ssd"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))),"efficientdet"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))),"sync_batchnorm"))
from utils.ssd.augmentations import SSDAugmentation
from utils.ssd.misc import *
from utils.ssd.box_utils import *