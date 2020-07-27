"""
各函数需要的工具函数，目前分开存放不同模型需要的工具，下一步参考mmdetection将工具函数进行集中和分类
"""

import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))),"ssd"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))),"efficientdet"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))),"sync_batchnorm"))
from utils.ssd.augmentations import SSDAugmentation
from utils.ssd.misc import *
from utils.ssd.box_utils import *