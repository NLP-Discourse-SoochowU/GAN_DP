# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
from config import *
from util.file_util import *


class RST:
    def __init__(self, train_set=TRAIN_SET_XL, dev_set=DEV_SET_XL, test_set=TEST_SET_XL):
        self.train = load_data(train_set)
        self.train_rst = load_data(RST_TRAIN_TREES_RST)
        self.dev = load_data(dev_set)
        self.dev_rst = load_data(RST_DEV_TREES_RST)
        self.test = load_data(test_set)
        self.test_rst = load_data(RST_TEST_TREES_RST)

        self.tree_obj_train = load_data(RST_TRAIN_TREES)
        self.tree_obj_dev = load_data(RST_DEV_TREES)
        self.tree_obj_test = load_data(RST_TEST_TREES)
        self.edu_tree_obj_test = load_data(RST_TEST_EDUS_TREES)

        if TRAIN_Ext:
            ext_trees = load_data(EXT_TREES)
            ext_trees_rst = load_data(EXT_TREES_RST)
            ext_set = load_data(EXT_SET_XL)
            self.train = self.train + ext_set
            self.train_rst = self.train_rst + ext_trees_rst
            self.tree_obj_train = self.tree_obj_train + ext_trees

    @staticmethod
    def get_voc_labels():
        word2ids = load_data(VOC_WORD2IDS_PATH)
        pos2ids = load_data(POS_word2ids_PATH)
        return word2ids, pos2ids

    @staticmethod
    def get_dev():
        return load_data(RST_DEV_TREES)

    @staticmethod
    def get_test():
        if USE_AE:
            return load_data(RST_TEST_EDUS_TREES)
        else:
            return load_data(RST_TEST_TREES)
