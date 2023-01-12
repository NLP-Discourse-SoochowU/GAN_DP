# coding: UTF-8
import torch
import numpy as np
from structure.nodes import Paragraph, Relation, rev_relationmap
import matplotlib
import matplotlib.pyplot as plt
from interface import ParserI
from structure.nodes import EDU, TEXT
from config import *


class PartPtrParser(ParserI):
    def __init__(self, model):
        self.parser = PartitionPtrParser(model)

    def parse(self, para):
        edus = []
        for edu in para.edus():
            edu_copy = EDU([TEXT(edu.text)])
            setattr(edu_copy, "words", edu.words)
            setattr(edu_copy, "tags", edu.tags)
            edus.append(edu_copy)
        return self.parser.parse(edus)


class PartitionPtrParser:
    def __init__(self, model):
        self.model = model

    def parse(self, edus, ret_session=False):
        if len(edus) < 2:
            return Paragraph(edus)

        # TODO implement beam search
        session = self.init_session(edus)
        while not session.terminate():
            split_scores, nr_score, state = self.decode(session)
            split = split_scores.argmax()
            nr_id = nr_score[split].argmax()
            nr = ids2nr[nr_id]
            nucl_id, rel_id = int(nr.split("-")[0]), int(nr.split("-")[1])
            nuclear = self.model.nuc_label.id2label[nucl_id]
            relation = self.model.rel_label.id2label[rel_id]
            session = session.forward(split_scores, state, split, nuclear, relation)
        root_relation = self.build_tree(edus, session.splits[:], session.nuclears[:], session.relations[:])
        discourse = Paragraph([root_relation])
        if ret_session:
            return discourse, session
        else:
            return discourse

    def init_session(self, edus):
        edu_words = [edu.words for edu in edus]
        edu_poses = [edu.tags for edu in edus]
        max_word_seqlen = max(len(words) for words in edu_words)
        edu_seqlen = len(edu_words)

        e_input_words = np.zeros((1, edu_seqlen, max_word_seqlen), dtype=np.long)
        e_boundaries = np.zeros([1, edu_seqlen + 1], dtype=np.long)
        e_input_poses = np.zeros_like(e_input_words)
        e_input_masks = np.zeros_like(e_input_words, dtype=np.uint8)

        for i, (words, poses) in enumerate(zip(edu_words, edu_poses)):
            e_input_words[0, i, :len(words)] = [self.model.word_vocab[word] for word in words]
            e_input_poses[0, i, :len(poses)] = [self.model.pos_vocab[pos] for pos in poses]
            e_input_masks[0, i, :len(words)] = 1
            if self.model.word_vocab[words[-1]] in [39, 3015, 178]:
                e_boundaries[0][i + 1] = 1

        e_input_words = torch.from_numpy(e_input_words).long()
        e_boundaries = torch.from_numpy(e_boundaries).long()
        e_input_poses = torch.from_numpy(e_input_poses).long()
        e_input_masks = torch.from_numpy(e_input_masks).byte()

        if self.model.use_gpu:
            e_input_words = e_input_words.cuda()
            e_boundaries = e_boundaries.cuda()
            e_input_poses = e_input_poses.cuda()
            e_input_masks = e_input_masks.cuda()

        edu_encoded, e_masks = self.model.encode_edus((e_input_words, e_input_poses, e_input_masks, e_boundaries))
        memory, _, context = self.model.encoder(edu_encoded, e_masks, e_boundaries)
        state = self.model.context_dense(context).unsqueeze(0)
        return Session(memory, state)

    def decode(self, session):
        left, right = session.stack[-1]
        return self.model(left, right, session.memory, session.state)

    def build_tree(self, edus, splits, nuclears, relations):
        left, split, right = splits.pop(0)
        nuclear = nuclears.pop(0)
        ftype = relations.pop(0)
        ctype = rev_relationmap[ftype]
        if split - left == 1:
            left_node = edus[left]
        else:
            left_node = self.build_tree(edus, splits, nuclears, relations)

        if right - split == 1:
            right_node = edus[split]
        else:
            right_node = self.build_tree(edus, splits, nuclears, relations)

        relation = Relation([left_node, right_node], nuclear=nuclear, ftype=ftype, ctype=ctype)
        return relation


class Session:
    def __init__(self, memory, state):
        self.n = memory.size(1) - 2
        self.step = 0
        self.memory = memory
        self.state = state
        self.stack = [(0, self.n + 1)]
        self.scores = np.zeros((self.n, self.n+2), dtype=np.float)
        self.splits = []
        self.nuclears = []
        self.relations = []

    def forward(self, score, state, split, nuclear, relation):
        left, right = self.stack.pop()
        if right - split > 1:
            self.stack.append((split, right))
        if split - left > 1:
            self.stack.append((left, split))
        self.splits.append((left, split, right))
        self.nuclears.append(nuclear)
        self.relations.append(relation)
        self.state = state
        self.scores[self.step] = score
        self.step += 1
        return self

    def terminate(self):
        return self.step >= self.n

    def draw_decision_hotmap(self):
        textcolors = ["black", "white"]
        cmap = "YlGn"
        ylabel = "split score"
        col_labels = ["split %d" % i for i in range(0, self.scores.shape[1])]
        row_labels = ["step %d" % i for i in range(1, self.scores.shape[0] + 1)]
        fig, ax = plt.subplots()
        im = ax.imshow(self.scores, cmap=cmap)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(ylabel, rotation=-90, va="bottom")
        ax.set_xticks(np.arange(self.scores.shape[1]))
        ax.set_yticks(np.arange(self.scores.shape[0]))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
        for edge, spine in ax.spines.items():
            spine.set_visible(False)
        ax.set_xticks(np.arange(self.scores.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(self.scores.shape[0] + 1) - .5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
        threshold = im.norm(self.scores.max()) / 2.
        valfmt = matplotlib.ticker.StrMethodFormatter("{x:.2f}")
        texts = []
        kw = dict(horizontalalignment="center", verticalalignment="center")
        for i in range(self.scores.shape[0]):
            for j in range(self.scores.shape[1]):
                kw.update(color=textcolors[im.norm(self.scores[i, j]) > threshold])
                text = im.axes.text(j, i, valfmt(self.scores[i, j], None), **kw)
                texts.append(text)
        fig.tight_layout()
        plt.show()

    def __repr__(self):
        return "[step %d]memory size: %s, state size: %s\n stack:\n%s\n, scores:\n %s" % \
               (self.step, str(self.memory.size()), str(self.state.size()),
                "\n".join(map(str, self.stack)) or "[]",
                str(self.scores))

    def __str__(self):
        return repr(self)
