# UTF-8
# Author: Longyin Zhang
# Date:

from util.file_util import *

SET = 65
nr2ids = load_data("data/nr2ids.pkl")
ids2nr = load_data("data/ids2nr.pkl")
SEED = 19  # 17, 7, 14, 2, 13, 19, 21, 22
NR_LABEL = 46
USE_GAN = True
MAX_H, MAX_W = 15, 25
in_channel_G, out_channel_G, ker_h_G, ker_w_G, strip_G = 2, 16, 3, MAX_W, 1
p_w_G, p_h_G = 3, 1
MAX_POOLING = True
WARM_UP_EP = 7 if USE_GAN else 50
LABEL_SWITCH, SWITCH_ITE = False, 4
USE_BOUND, BOUND_INFO_SIZE = False, 30
LAYER_NORM_USE = True
CONTEXT_ATT, ML_ATT_HIDDEN, HEADS = False, 128, 2

EDU_ENCODE_V = 1
Finetune_XL = True
XLNET_PATH = "data/pytorch_model.bin"
XLNET_SIZE, CHUNK_SIZE = 768, 512
