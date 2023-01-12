# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date: 2019.1.24
@Description: prepare the data needed.
"""
import torch
from config import nr2ids, nucl2ids, coarse2ids
from path_config import *
from util.file_util import *


def form_set_xl():
    build_one(RST_TRAIN_TREES, TRAIN_SET_NEG, "xl")
    build_one(RST_DEV_TREES, DEV_SET_NEG, "xl")
    build_one(RST_TEST_TREES, TEST_SET_NEG, "xl")
    build_one(EXT_TREES, EXT_SET_NEG, "xl")


def form_set():
    build_one(RST_TRAIN_TREES, TRAIN_SET)
    build_one(RST_DEV_TREES, DEV_SET)
    build_one(RST_TEST_TREES, TEST_SET)
    build_one(EXT_TREES, EXT_SET)


def build_one(src_, dist_, type_=None):
    trees = load_data(src_)
    instances = []
    for tree in trees:
        encoder_inputs, decoder_inputs = [], []
        pred_splits, pred_nr = [], []
        pred_n, pred_r = [], []
        edus = list(tree.edus)
        for edu in edus:
            edu_word_ids = edu.temp_edu_ids
            edu_pos_ids = edu.temp_pos_ids
            edu_elmo_embeddings = edu.temp_edu_emlo_emb
            # boundary
            tmp_line = edu.temp_line.strip()
            if not tmp_line.endswith("_!) )") and not tmp_line.endswith("_!) )//TT_ERR"):
                input("pp_dt_43")
                bound_info = None
            elif tmp_line.endswith("<P>_!) )") or tmp_line.endswith("<P>_!) )//TT_ERR"):
                bound_info = 2
            elif tmp_line.endswith("._!) )") or tmp_line.endswith("._!) )//TT_ERR"):
                bound_info = 1
            else:
                bound_info = 0
            if type_ == "xl":
                input(edu.temp_edu)
                encoder_inputs.append((edu.temp_edu, edu_word_ids, edu_elmo_embeddings, edu_pos_ids, bound_info))
            else:
                encoder_inputs.append((edu_word_ids, edu_elmo_embeddings, edu_pos_ids, bound_info))
        splits = gen_decoder_data(tree)
        graph_s, graph_n, graph_r, graph_nr, line_idx = [], [], [], [], [0]
        len_ = splits[0][2] - splits[0][0] + 1
        idx = 0
        queue_ = []
        for start, split, end, nuc, rel in splits:
            decoder_inputs.append((start, end))
            pred_splits.append(split)
            pred_nr.append(nr2ids[nuc + "-" + rel])
            pred_n.append(nucl2ids[nuc])
            pred_r.append(coarse2ids[rel])
            # build tmp_tri_fig
            if len(queue_) == 0:  # initialization
                queue_.append(start)
                graph_s.append([-1 for _ in range(len_)])
                graph_n.append([0 for _ in range(len_)])
                graph_r.append([0 for _ in range(len_)])
                graph_nr.append([0 for _ in range(len_)])
                # queue_.append((split, idx))
                graph_s[idx][split] = 0
                graph_n[idx][split] = (pred_n[-1] + 1)
                graph_r[idx][split] = (pred_r[-1] + 1)
                graph_nr[idx][split] = (pred_nr[-1] + 1)
            else:
                idx += 1
                if start < queue_[-1]:
                    while start < queue_[-1]:
                        queue_.pop(-1)
                        idx -= 1
                if idx >= len(graph_s):
                    graph_s.append([-1 for _ in range(len_)])
                    graph_n.append([0 for _ in range(len_)])
                    graph_r.append([0 for _ in range(len_)])
                    graph_nr.append([0 for _ in range(len_)])
                queue_.append(start)
                # queue_.append((split, idx))
                graph_s[idx][split] = 0
                graph_n[idx][split] = (pred_n[-1] + 1)
                graph_r[idx][split] = (pred_r[-1] + 1)
                graph_nr[idx][split] = (pred_nr[-1] + 1)
                line_idx.append(idx)

            A = split - start == 1
            B = end - split == 1
            if A or B:
                if idx + 1 >= len(graph_s):
                    graph_s.append([-2 for _ in range(len_)])
                    graph_n.append([0 for _ in range(len_)])
                    graph_r.append([0 for _ in range(len_)])
                    graph_nr.append([0 for _ in range(len_)])
            if A:
                graph_s[idx + 1][split - 1] = -1
                graph_nr[idx + 1][split - 1] = -1
            if B:
                graph_s[idx + 1][split + 1] = -1
                graph_nr[idx + 1][split + 1] = -1

        instances.append((encoder_inputs, decoder_inputs, pred_splits, pred_nr, graph_s,
                          graph_n, graph_r, graph_nr, line_idx))
    save_data(instances, dist_)


def gen_decoder_data(root):
    """
                splits    s0  s1  s2  s3  s4  s5  s6
                edus    s/  e0  e1  e2  e3  e4  e5  /s
    """
    splits = []  # [(0, 3, 6, NS), (0, 2, 3, SN), ...]
    buffer_ = [(idx, idx) for idx in range(1, len(root.edus) + 1)]
    stack_ = []
    for node_ in root.nodes:
        if node_.left_child is None and node_.right_child is None:
            tmp_edu_span = buffer_.pop(0)
            stack_.append(tmp_edu_span)
        else:
            # reduce
            right_span = stack_.pop()
            left_span = stack_.pop()
            stack_.append((left_span[0], right_span[1]))
            # split
            splits.append((left_span[0] - 1, left_span[1], right_span[1], node_.child_NS_rel, node_.child_rel))
    splits = splits[::-1]
    return splits


def fetch_edus():
    tr_trees, dev_trees = load_data(RST_TRAIN_TREES), load_data(RST_DEV_TREES)
    tr_edus, dev_edus = [], []
    for tree_ in tr_trees:
        edus = tree_.edus
        for edu in edus:
            tr_edus.append(edu.temp_edu)
    for tree_ in dev_trees:
        edus = tree_.edus
        for edu in edus:
            dev_edus.append(edu.temp_edu)
    write_iterate(tr_edus, RST_EDUs_ori_tr)
    write_iterate(dev_edus, RST_EDUs_ori_de)


def transform_edus_tree_elmo_from_gold():
    gold_test = load_data(RST_TEST_TREES)
    ori_test = load_data(RST_TEST_EDUS_TREES_)
    s_trees = []
    for g_tree, o_tree in zip(gold_test, ori_test):
        # check each tree
        tree_elmo_li = None
        for edu in g_tree.edus:
            tree_elmo_li = edu.temp_edu_emlo_emb if tree_elmo_li is None \
                else torch.cat((tree_elmo_li, edu.temp_edu_emlo_emb), 0)
        # (edu_num, hidden)
        begin = 0
        result_edus = []
        for edu in o_tree.edus:
            edu_len = len(edu.temp_edu_ids)
            tmp_edu_emb = tree_elmo_li[begin: begin + edu_len]
            edu.temp_edu_emlo_emb = tmp_edu_emb
            result_edus.append(edu)
            begin += edu_len
        o_tree.edus = result_edus
        s_trees.append(o_tree)
    save_data(s_trees, RST_TEST_EDUS_TREES_)
    # all_tree = gold_test + ori_test
    # for tree in all_tree:
    #     for edu in tree.edus:
    #         a = len(edu.temp_edu_ids)
    #         b = edu.temp_edu_emlo_emb.size(0)
    #         if a != b:
    #             input("fuck")


def transform_tf2bound():
    a = load_data(RST_TEST_EDUS_TREES)  # to change
    a_new = []
    b = load_data(RST_TEST_TREES)  # gold
    for tree_a, tree_b in zip(a, b):
        print(len(tree_a.edus), len(tree_b.edus))
        edus_a, edus_b = tree_a.edus, tree_b.edus
        edus_a_new = []
        edu_num = len(edus_a)
        edu_num_b = len(edus_b)
        for idx in range(edu_num):
            edu_a = edus_a[idx]
            auto_edu = edu_a.temp_edu.strip().split()[-1].lower()
            bound_info = 0
            for idx_ in range(edu_num_b):
                edu_b = edus_b[idx_]
                tmp_edu_end = edu_b.temp_edu.strip().split()[-1].lower()
                if auto_edu == tmp_edu_end:
                    gold_line = edu_b.temp_line.strip()
                    if not gold_line.endswith("_!) )") and not gold_line.endswith("_!) )//TT_ERR"):
                        input(tmp_line)
                        bound_info = 0
                    elif gold_line.endswith("<P>_!) )") or gold_line.endswith("<P>_!) )//TT_ERR"):
                        bound_info = 2
                    elif gold_line.endswith("._!) )") or gold_line.endswith("._!) )//TT_ERR"):
                        bound_info = 1
                    else:
                        bound_info = 0
                    break
            edu_a.edu_node_boundary = bound_info
            edus_a_new.append(edu_a)
        tree_a.edus = edus_a_new
        a_new.append(tree_a)
    save_data(a_new, RST_TEST_EDUS_TREES_)


if __name__ == "__main__":
    # dt_builder = Builder()
    # dt_builder.form_trees()
    # fetch_edus()
    # dt_builder.extent_trees()
    # dt_builder.update_edu_trees()
    # form_label2ids()
    form_set()
