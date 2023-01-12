# coding: UTF-8
import numpy as np
from config import ids2nr
from structure.rst_tree import rst_tree


class PartitionPtrParser:
    def __init__(self, model):
        self.model = model
        self.ids2nr = ids2nr

    def parse(self, edus, ret_session=False):
        session = self.model.init_session(edus)
        while not session.terminate():
            split_score, nr_score, state = self.model(session)
            split = split_score.argmax()
            nr = self.ids2nr[nr_score[split].argmax()]
            nuclear, relation = nr.split("-")[0], "-".join(nr.split("-")[1:])
            session = session.forward(split_score, state, split, nuclear, relation)
        tree_parsed = self.build_rst_tree(edus, session.splits[:], session.nuclear[:], session.relations[:])
        if ret_session:
            return tree_parsed, session
        else:
            return tree_parsed

    def build_rst_tree(self, edus, splits, nuclear, relations):
        left, split, right = splits.pop(0)
        nucl = nuclear.pop(0)
        rel = relations.pop(0)
        if right - split == 1:
            # leaf node
            right_node = edus[split]
        else:
            # non leaf
            right_node = self.build_rst_tree(edus, splits, nuclear, relations)
        if split - left == 1:
            # leaf node
            left_node = edus[left]
        else:
            # none leaf
            left_node = self.build_rst_tree(edus, splits, nuclear, relations)
        root = rst_tree(l_ch=left_node, r_ch=right_node, ch_ns_rel=nucl, child_rel=rel,
                        temp_edu_span=(left_node.temp_edu_span[0], right_node.temp_edu_span[1]))
        return root

    def traverse_tree(self, root):
        if root.left_child is not None:
            print(root.temp_edu_span, ", ", root.child_NS_rel, ", ", root.child_rel)
            self.traverse_tree(root.right_child)
            self.traverse_tree(root.left_child)

    def draw_scores_matrix(self):
        scores = self.model.scores
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
