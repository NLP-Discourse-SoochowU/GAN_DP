# coding: UTF-8

import logging
from collections import defaultdict
from gensim.models import KeyedVectors
import numpy as np
import torch


logger = logging.getLogger(__name__)
PAD_TAG = "<pad>"
UNK_TAG = "<unk>"


class Vocab:
    def __init__(self, name, counter, min_occur=1):
        self.name = name
        self.s2id = defaultdict(int)
        self.s2id[UNK_TAG] = 0
        self.id2s = [UNK_TAG]

        self.counter = counter
        for tag, freq in counter.items():
            if tag not in self.s2id and freq >= min_occur:
                self.s2id[tag] = len(self.s2id)
                self.id2s.append(tag)
        logger.info("%s vocabulary size %d" % (self.name, len(self.s2id)))

    def __getitem__(self, item):
        return self.s2id[item]

    def embedding(self, dim=None, pretrained=None, binary=False, freeze=False, use_gpu=False):
        if dim is not None and pretrained is not None:
            raise Warning("dim should not given if pretraiained weights are assigned")

        if dim is None and pretrained is None:
            raise Warning("one of dim or pretrained should be assigned")

        if pretrained:
            w2v = KeyedVectors.load_word2vec_format(pretrained, binary=binary)
            dim = w2v.vector_size
            scale = np.sqrt(3.0 / dim)
            weights = np.empty([len(self), dim], dtype=np.float32)
            oov_count = 0
            all_count = 0
            for tag, i in self.s2id.items():
                if tag in w2v.vocab:
                    weights[i] = w2v[tag].astype(np.float32)
                else:
                    oov_count += self.counter[tag]
                    weights[i] = np.zeros(dim).astype(np.float32) if freeze else \
                        np.random.uniform(-scale, scale, dim).astype(np.float32)
                all_count += self.counter[tag]
            logger.info("%s vocabulary pretrained OOV %d/%d, %.2f%%" %
                        (self.name, oov_count, all_count, oov_count/all_count*100))
        else:
            scale = np.sqrt(3.0 / dim)
            weights = np.random.uniform(-scale, scale, [len(self), dim]).astype(np.float32)
        weights[0] = np.zeros(dim).astype(np.float32) if freeze else \
            np.random.uniform(-scale, scale, dim).astype(np.float32)
        weights = torch.from_numpy(weights)
        if use_gpu:
            weights = weights.cuda()
        embedding = torch.nn.Embedding.from_pretrained(weights, freeze=freeze)
        return embedding

    def __len__(self):
        return len(self.id2s)


class Label:
    def __init__(self, name, counter, specials=None):
        self.name = name
        self.counter = counter.copy()
        self.label2id = {}
        self.id2label = []
        if specials:
            for label in specials:
                del self.counter[label]
                self.label2id[label] = len(self.label2id)
                self.id2label.append(label)

        for label, freq in self.counter.items():
            if label not in self.label2id:
                self.label2id[label] = len(self.label2id)
                self.id2label.append(label)
        logger.info("label %s size %d" % (name, len(self)))

    def __getitem__(self, item):
        return self.label2id[item]

    def __len__(self):
        return len(self.id2label)
