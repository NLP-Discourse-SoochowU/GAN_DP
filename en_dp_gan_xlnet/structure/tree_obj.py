# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date: 2018/4/5
@Description:   tree.edus in the form of ids
                tree.nodes in the form of rst_tree objective
"""
import copy


class tree_obj:
    def __init__(self, tree=None):
        self.edus = list()
        self.nodes = list()
        self.sents_edus = list()
        if tree is not None:
            self.file_name = tree.file_name
            # init by tree
            self.tree = tree
            self.pre_traverse(tree)
            # self.get_sents_edus()

    def __copy__(self):
        tree_ = copy.copy(self.tree)
        edus_ = [copy.copy(edu) for edu in self.edus]
        nodes_ = [copy.copy(node) for node in self.nodes]
        t_o = tree_obj(tree_)
        t_o.edus = edus_
        t_o.nodes = nodes_
        return t_o

    def assign_edus(self, edus_list):
        for edu in edus_list:
            self.edus.append(edu)

    def pre_traverse(self, root):
        if root is None:
            return
        self.pre_traverse(root.left_child)
        self.pre_traverse(root.right_child)
        if root.left_child is None and root.right_child is None:
            self.edus.append(root)
            self.nodes.append(root)
        else:
            self.nodes.append(root)

    def get_sents_edus(self):
        tmp_sent_edus = list()
        for edu in self.edus:
            tmp_sent_edus.append(edu)
            if edu.edu_node_boundary:
                self.sents_edus.append(tmp_sent_edus)
                tmp_sent_edus = list()
