# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date: 2018.4.5
@Description:
"""
import progressbar
from structure.tree_obj import tree_obj
from stanfordcorenlp import StanfordCoreNLP
from config import *
from path_config import path_to_jar
from structure.rst_tree import rst_tree
from util.rst_utils import get_edus_info


class Builder:
    def __init__(self):
        self.nlp = StanfordCoreNLP(path_to_jar)
        self.max_len_of_edu = 0

    @staticmethod
    def update_edu_trees():
        edu_trees = load_data(RST_TEST_EDUS_TREES)
        test_trees = load_data(RST_TEST_TREES)
        tree_num = len(test_trees)
        new_edu_trees = []
        for idx in range(tree_num):
            tmp_gold_tree = test_trees[idx]
            tmp_token_vec = []
            for edu in tmp_gold_tree.edus:
                tmp_token_vec.extend(edu.temp_edu_emlo_emb)
            tmp_edu_tree = edu_trees[idx]
            new_edus = []
            for edu in tmp_edu_tree.edus:
                edu_len = len(edu.temp_edu_ids)
                edu.temp_edu_emlo_emb = tmp_token_vec[:edu_len]
                tmp_token_vec = tmp_token_vec[edu_len:]
                new_edus.append(edu)
            new_tree = tree_obj()
            new_tree.assign_edus(new_edus)
            new_edu_trees.append(new_tree)
        save_data(new_edu_trees, RST_TEST_EDUS_TREES)

    def form_trees(self):
        # train_dir = os.path.join(RAW_RST_DT_PATH, "train")
        # train_tree_p = os.path.join(RST_DT_PATH, "train_trees.pkl")
        # train_trees_rst_p = os.path.join(RST_DT_PATH, "train_trees_rst.pkl")
        # self.build_specific_trees(train_dir, train_tree_p, train_trees_rst_p)
        #
        # dev_dir = os.path.join(RAW_RST_DT_PATH, "dev")
        # dev_tree_p = os.path.join(RST_DT_PATH, "dev_trees.pkl")
        # dev_trees_rst_p = os.path.join(RST_DT_PATH, "dev_trees_rst.pkl")
        # self.build_specific_trees(dev_dir, dev_tree_p, dev_trees_rst_p)

        test_dir = os.path.join(RAW_RST_DT_PATH, "test")
        test_tree_p = os.path.join(RST_DT_PATH, "test_trees.pkl")
        test_trees_rst_p = os.path.join(RST_DT_PATH, "test_trees_rst.pkl")
        edus_tree_p = os.path.join(RST_DT_PATH, "edus_test_trees.pkl")
        self.build_specific_trees(test_dir, test_tree_p, test_trees_rst_p, edus_tree_p)

        # test_dir = os.path.join(RAW_RST_DT_PATH, "test")
        # test_tree_p_n = os.path.join(RST_DT_PATH, "test_trees_n.pkl")
        # test_trees_rst_p_n = os.path.join(RST_DT_PATH, "test_trees_rst_n.pkl")
        # edus_tree_p_n = os.path.join(RST_DT_PATH, "edus_test_trees_n.pkl")
        # self.build_specific_trees(test_dir, test_tree_p_n, test_trees_rst_p_n, edus_tree_p_n)

    def build_specific_trees(self, raw_p=None, tree_p=None, tree_rst_p=None, edus_tree_p=None):
        p = progressbar.ProgressBar()
        p.start(len(os.listdir(raw_p)))
        pro_idx = 1
        trees_list, trees_rst_list = [], []
        edu_trees_list = []
        for file_name in os.listdir(raw_p):
            p.update(pro_idx)
            pro_idx += 1
            if file_name.endswith('.out.dis'):
                root, edus_list = self.build_one_tree(raw_p, file_name, edus_tree_p)
                trees_rst_list.append(root)
                tree_obj_ = tree_obj(root)
                trees_list.append(tree_obj_)
                if edus_tree_p is not None:
                    edus_tree_obj_ = tree_obj()
                    edus_tree_obj_.assign_edus(edus_list)
                    edu_trees_list.append(edus_tree_obj_)
        p.finish()
        save_data(trees_list, tree_p)
        save_data(trees_rst_list, tree_rst_p)
        if edus_tree_p is not None:
            save_data(edu_trees_list, edus_tree_p)

    def build_one_tree(self, raw_p, file_name, edus_tree_p):
        """ build edu_list and edu_ids according to EDU files.
        """
        temp_path = os.path.join(raw_p, file_name)
        e_bound_list, e_list, e_span_list, e_ids_list, e_tags_list, e_headwords_list, e_cent_word_list, e_emlo_list =\
            get_edus_info(temp_path.replace(".dis", ".edus"), temp_path.replace(".out.dis", ".out"), nlp=self.nlp)
        lines_list = open(temp_path, 'r').readlines()
        root = rst_tree(type_="Root", lines_list=lines_list, temp_line=lines_list[0], file_name=file_name, rel="span")
        root.create_tree(temp_line_num=1, p_node_=root)
        root.config_edus(temp_node=root, e_list=e_list, e_span_list=e_span_list, e_ids_list=e_ids_list,
                         e_tags_list=e_tags_list, e_headwords_list=e_headwords_list, e_cent_word_list=e_cent_word_list,
                         total_e_num=len(e_ids_list), e_bound_list=e_bound_list, e_emlo_list=e_emlo_list)

        edus_list = []
        if edus_tree_p is not None:
            e_bound_list, e_list, e_span_list, e_ids_list, e_tags_list, e_headwords_list, e_cent_word_list, \
                e_emlo_list = get_edus_info(temp_path.replace(".dis", ".auto.edus"),
                                            temp_path.replace(".out.dis", ".out"), nlp=self.nlp)
            edu_num = len(e_bound_list)
            for idx in range(edu_num):
                tmp_edu = rst_tree(temp_edu_boundary=e_bound_list[idx], temp_edu=e_list[idx],
                                   temp_edu_span=e_span_list[idx], temp_edu_ids=e_ids_list[idx],
                                   temp_pos_ids=e_tags_list[idx], temp_edu_heads=e_headwords_list[idx],
                                   temp_edu_has_center_word=e_cent_word_list[idx],
                                   tmp_edu_emlo_emb=e_emlo_list[idx])
                edus_list.append(tmp_edu)
        return root, edus_list

    def build_tree_obj_list(self):
        raw = ""
        parse_trees = []
        for file_name in os.listdir(raw):
            if file_name.endswith(".out"):
                tmp_edus_list = []
                sent_path = os.path.join(raw, file_name)
                edu_path = sent_path + ".edu"
                edus_boundary_list, edus_list, edu_span_list, edus_ids_list, edus_tag_ids_list, edus_conns_list, \
                    edu_headword_ids_list, edu_has_center_word_list = get_edus_info(edu_path, sent_path, nlp=self.nlp)
                for _ in range(len(edus_list)):
                    tmp_edu = rst_tree()
                    tmp_edu.temp_edu = edus_list.pop(0)
                    tmp_edu.temp_edu_span = edu_span_list.pop(0)
                    tmp_edu.temp_edu_ids = edus_ids_list.pop(0)
                    tmp_edu.temp_pos_ids = edus_tag_ids_list.pop(0)
                    tmp_edu.temp_edu_conn_ids = edus_conns_list.pop(0)
                    tmp_edu.temp_edu_heads = edu_headword_ids_list.pop(0)
                    tmp_edu.temp_edu_has_center_word = edu_has_center_word_list.pop(0)
                    tmp_edu.edu_node_boundary = edus_boundary_list.pop(0)
                    tmp_edu.inner_sent = True
                    tmp_edus_list.append(tmp_edu)
                tmp_tree_obj = tree_obj()
                tmp_tree_obj.file_name = file_name
                tmp_tree_obj.assign_edus(tmp_edus_list)
                parse_trees.append(tmp_tree_obj)
        return parse_trees
