# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
# RST-DT
RST_DT_PATH = "data/rst_dt"
RAW_RST_DT_PATH = "data/rst_dt/RST_DT_RAW"

# EDUs of train & dev
RST_EDUs_ori_tr = "data/EDUs/ori_train_edus.tsv"
RST_EDUs_ori_de = "data/EDUs/ori_dev_edus.tsv"
# EN EDUs of train & dev
RST_EDUs_google_tr = "data/EDUs/translate_train.txt"
RST_EDUs_google_de = "data/EDUs/translate_dev.tsv"

EXT_TREES = "data/rst_dt/ext_trees.pkl"
EXT_TREES_RST = "data/rst_dt/ext_trees_rst.pkl"
EXT_SET = "data/rst_dt/ext_set.pkl"

# RST-DT Generated
RST_TRAIN_TREES = "data/rst_dt/train_trees.pkl"
RST_TRAIN_TREES_RST = "data/rst_dt/train_trees_rst.pkl"
TRAIN_SET = "data/rst_dt/train_set.pkl"
RST_TEST_TREES = "data/rst_dt/test_trees.pkl"
RST_TEST_TREES_RST = "data/rst_dt/test_trees_rst.pkl"
RST_TEST_EDUS_TREES = "data/rst_dt/edus_test_trees.pkl"
RST_TEST_EDUS_TREES_ = "data/rst_dt/edus_test_trees_.pkl"
TEST_SET = "data/rst_dt/test_set.pkl"
RST_DEV_TREES = "data/rst_dt/dev_trees.pkl"
RST_DEV_TREES_RST = "data/rst_dt/dev_trees_rst.pkl"
DEV_SET = "data/rst_dt/dev_set.pkl"
TRAIN_SET_XL = "data/rst_dt/train_set_xl.pkl"
TEST_SET_XL = "data/rst_dt/test_set_xl.pkl"
DEV_SET_XL = "data/rst_dt/dev_set_xl.pkl"
EXT_SET_XL = "data/rst_dt/ext_set_xl.pkl"

TRAIN_SET_XLM = "data/rst_dt/marcu/train_set_xlm.pkl"  # marcu 学习
TEST_SET_XLM = "data/rst_dt/marcu/test_set_xlm.pkl"
DEV_SET_XLM = "data/rst_dt/marcu/dev_set_xlm.pkl"
EXT_SET_XLM = "data/rst_dt/marcu/ext_set_xlm.pkl"
NR2ids_marcu = "data/rst_dt/marcu/nr2ids.pkl"
IDS2nr_marcu = "data/rst_dt/marcu/ids2nr.pkl"

TRAIN_SET_NR = "data/rst_dt/nr_train_set.pkl"  # N 和 R 分开
TEST_SET_NR = "data/rst_dt/nr_test_set.pkl"
DEV_SET_NR = "data/rst_dt/nr_dev_set.pkl"

RST_TEST_TREES_N = "data/rst_dt/test_trees_n.pkl"
RST_TEST_TREES_RST_N = "data/rst_dt/test_trees_rst_n.pkl"
RST_TEST_EDUS_TREES_N = "data/rst_dt/edus_test_trees_n.pkl"
TEST_SET_N = "data/rst_dt/test_n.set.pkl"
TEST_SET_NR_N = "data/rst_dt/nr_test_n.set.pkl"

# VOC
REL_raw2coarse = "data/voc/raw2coarse.pkl"
VOC_WORD2IDS_PATH = "data/voc/word2ids.pkl"
POS_word2ids_PATH = "data/voc/pos2ids.pkl"
REL_coarse2ids = "data/voc/coarse2ids.pkl"
REL_ids2coarse = "data/voc/ids2coarse.pkl"
VOC_VEC_PATH = "data/voc/ids2vec.pkl"
LABEL2IDS = "data/voc/label2ids.pkl"
IDS2LABEL = "data/voc/ids2label.pkl"

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/" \
               "elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo" \
              "_2x4096_512_2048cnn_2xhighway_weights.hdf5"
path_to_jar = 'stanford-corenlp-full-2018-02-27'
MODELS2SAVE = "data/models_saved"
MODEL_SAVE = "data/models/treebuilder.partptr.basic"
LOG_ALL = "data/logs"
PRE_DEPTH = "data/pre_depth.pkl."
GOLD_DEPTH = "data/gold_depth.pkl"
DRAW_DT = "data/drawer.pkl"
LOSS_PATH = "data/loss/"

# ADDITIONAL
EXT_SET_NEG = "data/rst_dt/neg_ext_set.pkl"
TRAIN_SET_NEG = "data/rst_dt/neg_train_set.pkl"
TEST_SET_NEG = "data/rst_dt/neg_test_set.pkl"
DEV_SET_NEG = "data/rst_dt/neg_dev_set.pkl"
