# coding: UTF-8
import numpy as np
import torch
from config import ids2nr, SEED
from structure.rst_tree import rst_tree
import random
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


class PartitionPtrParser:
    def __init__(self, model):
        self.ids2nr = ids2nr

    def parse(self, edus, ret_session=False, model=None, model_xl=None, tokenizer_xl=None):
        # TODO implement beam search
        session = model.init_session(edus, model_xl, tokenizer_xl)  # 初始化状态
        d_masks, splits = None, []
        while not session.terminate():
            split_score, nr_score, state, d_mask = model.parse_predict(session)
            d_masks = d_mask if d_masks is None else torch.cat((d_masks, d_mask), 1)
            split = split_score.argmax()
            nr = self.ids2nr[nr_score[split].argmax()]
            nuclear, relation = nr.split("-")[0], "-".join(nr.split("-")[1:])
            session = session.forward(split_score, state, split, nuclear, relation)
        # build tree by splits (left, split, right)
        tree_parsed = self.build_rst_tree(edus, session.splits[:], session.nuclear[:], session.relations[:])
        if ret_session:
            return tree_parsed, session
        else:
            return tree_parsed

    def build_rst_tree(self, edus, splits, nuclear, relations, type_="Root", rel_=None):
        left, split, right = splits.pop(0)
        nucl = nuclear.pop(0)
        rel = relations.pop(0)
        left_n, right_n = nucl[0], nucl[1]
        left_rel = rel if left_n == "N" else "span"
        right_rel = rel if right_n == "N" else "span"
        if right - split == 0:
            # leaf node
            right_node = edus[split + 1]
        else:
            # non leaf
            right_node = self.build_rst_tree(edus, splits, nuclear, relations, type_=right_n, rel_=right_rel)
        if split - left == 0:
            # leaf node
            left_node = edus[split]
        else:
            # none leaf
            left_node = self.build_rst_tree(edus, splits, nuclear, relations, type_=left_n, rel_=left_rel)
        node_height = max(left_node.node_height, right_node.node_height) + 1
        root = rst_tree(l_ch=left_node, r_ch=right_node, ch_ns_rel=nucl, child_rel=rel,
                        temp_edu_span=(left_node.temp_edu_span[0], right_node.temp_edu_span[1]),
                        node_height=node_height, type_=type_, rel=rel_)
        return root

    def traverse_tree(self, root):
        if root.left_child is not None:
            print(root.temp_edu_span, ", ", root.child_NS_rel, ", ", root.child_rel)
            self.traverse_tree(root.right_child)
            self.traverse_tree(root.left_child)

    def draw_scores_matrix(self, model):
        scores = model.scores
        self.draw_decision_hot_map(scores)

    @staticmethod
    def draw_decision_hot_map(scores):
        import matplotlib
        import matplotlib.pyplot as plt
        text_colors = ["black", "white"]
        c_map = "YlGn"
        y_label = "split score"
        col_labels = ["split %d" % i for i in range(0, scores.shape[1])]
        row_labels = ["step %d" % i for i in range(1, scores.shape[0] + 1)]
        fig, ax = plt.subplots()
        im = ax.imshow(scores, cmap=c_map)
        c_bar = ax.figure.colorbar(im, ax=ax)
        c_bar.ax.set_ylabel(y_label, rotation=-90, va="bottom")
        ax.set_xticks(np.arange(scores.shape[1]))
        ax.set_yticks(np.arange(scores.shape[0]))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
        for edge, spine in ax.spines.items():
            spine.set_visible(False)
        ax.set_xticks(np.arange(scores.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(scores.shape[0] + 1) - .5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
        threshold = im.norm(scores.max()) / 2.
        val_fmt = matplotlib.ticker.StrMethodFormatter("{x:.2f}")
        texts = []
        kw = dict(horizontalalignment="center", verticalalignment="center")
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                kw.update(color=text_colors[im.norm(scores[i, j]) > threshold])
                text = im.axes.text(j, i, val_fmt(scores[i, j], None), **kw)
                texts.append(text)
        fig.tight_layout()
        plt.show()
