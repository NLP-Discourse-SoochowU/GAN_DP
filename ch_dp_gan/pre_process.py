# UTF-8
# Author: Longyin Zhang
# Date: 2020.10.9
import argparse
import logging
import numpy as np
from dataset import CDTB
from collections import Counter
from itertools import chain
from structure.vocab import Vocab, Label
from structure.nodes import node_type_filter, EDU, Relation, Sentence, TEXT
import progressbar
from util.file_util import *
p = progressbar.ProgressBar()


def build_vocab(dataset):
    word_freq = Counter()
    pos_freq = Counter()
    nuc_freq = Counter()
    rel_freq = Counter()
    for paragraph in chain(*dataset):
        for node in paragraph.iterfind(filter=node_type_filter([EDU, Relation])):
            if isinstance(node, EDU):
                word_freq.update(node.words)
                pos_freq.update(node.tags)
            elif isinstance(node, Relation):
                nuc_freq[node.nuclear] += 1
                rel_freq[node.ftype] += 1

    word_vocab = Vocab("word", word_freq)
    pos_vocab = Vocab("part of speech", pos_freq)
    nuc_label = Label("nuclear", nuc_freq)
    rel_label = Label("relation", rel_freq)
    return word_vocab, pos_vocab, nuc_label, rel_label


def gen_decoder_data(root, edu2ids):
    # splits    s0  s1  s2  s3  s4  s5  s6
    # edus    s/  e0  e1  e2  e3  e4  e5  /s
    splits = []  # [(0, 3, 6, NS), (0, 2, 3, SN), ...]
    child_edus = []  # [edus]

    if isinstance(root, EDU):
        child_edus.append(root)
    elif isinstance(root, Sentence):
        for child in root:
            _child_edus, _splits = gen_decoder_data(child, edu2ids)
            child_edus.extend(_child_edus)
            splits.extend(_splits)
    elif isinstance(root, Relation):
        children = [gen_decoder_data(child, edu2ids) for child in root]
        if len(children) < 2:
            raise ValueError("relation node should have at least 2 children")

        while children:
            left_child_edus, left_child_splits = children.pop(0)
            if children:
                last_child_edus, _ = children[-1]
                start = edu2ids[left_child_edus[0]]
                split = edu2ids[left_child_edus[-1]] + 1
                end = edu2ids[last_child_edus[-1]] + 1
                nuc = root.nuclear
                rel = root.ftype
                splits.append((start, split, end, nuc, rel))
            child_edus.extend(left_child_edus)
            splits.extend(left_child_splits)
    return child_edus, splits


def numericalize(dataset, word_vocab, pos_vocab, nuc_label, rel_label):
    instances = []
    for paragraph in filter(lambda d: d.root_relation(), chain(*dataset)):
        encoder_inputs = []
        decoder_inputs = []
        pred_splits = []
        pred_nucs = []
        pred_rels = []
        edus = list(paragraph.edus())
        for edu in edus:
            edu_word_ids = [word_vocab[word] for word in edu.words]
            edu_pos_ids = [pos_vocab[pos] for pos in edu.tags]
            encoder_inputs.append((edu_word_ids, edu_pos_ids))
        edu2ids = {edu: i for i, edu in enumerate(edus)}
        _, splits = gen_decoder_data(paragraph.root_relation(), edu2ids)
        for start, split, end, nuc, rel in splits:
            decoder_inputs.append((start, end))
            pred_splits.append(split)
            pred_nucs.append(nuc_label[nuc])
            pred_rels.append(rel_label[rel])
        instances.append((encoder_inputs, decoder_inputs, pred_splits, pred_nucs, pred_rels))
    return instances


nr2ids = dict()
ids2nr = dict()
nr_idx = 0


def gen_batch_iter(instances, batch_size, use_gpu=False):
    global nr2ids, nr_idx, ids2nr
    random_instances = np.random.permutation(instances)
    num_instances = len(instances)
    offset = 0
    p.start(num_instances)
    while offset < num_instances:
        p.update(offset)
        batch = random_instances[offset: min(num_instances, offset+batch_size)]
        for batchi, (_, _, _, pred_nucs, pred_rels) in enumerate(batch):
            for nuc, rel in zip(pred_nucs, pred_rels):
                nr = str(nuc) + "-" + str(rel)
                if nr not in nr2ids.keys():
                    nr2ids[nr] = nr_idx
                    ids2nr[nr_idx] = nr
                    nr_idx += 1
        offset = offset + batch_size
    p.finish()


def main(args):
    cdtb = CDTB(args.data, "TRAIN", "VALIDATE", "TEST", ctb_dir=args.ctb_dir, preprocess=True, cache_dir=args.cache_dir)
    word_vocab, pos_vocab, nuc_label, rel_label = build_vocab(cdtb.train)
    trainset = numericalize(cdtb.train, word_vocab, pos_vocab, nuc_label, rel_label)
    test = numericalize(cdtb.test, word_vocab, pos_vocab, nuc_label, rel_label)
    dev = numericalize(cdtb.validate, word_vocab, pos_vocab, nuc_label, rel_label)
    for idx in range(1, 8):
        gen_batch_iter(trainset, args.batch_size, args.use_gpu)
        gen_batch_iter(test, args.batch_size, args.use_gpu)
        gen_batch_iter(dev, args.batch_size, args.use_gpu)
    print(nr2ids)
    save_data(nr2ids, "data/nr2ids.pkl")
    save_data(ids2nr, "data/ids2nr.pkl")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_parser = argparse.ArgumentParser()

    # dataset parameters
    arg_parser.add_argument("--data", default="data/CDTB")
    arg_parser.add_argument("--ctb_dir", default="data/CTB")
    arg_parser.add_argument("--cache_dir", default="data/cache")

    # model parameters
    arg_parser.add_argument("-hidden_size", default=512, type=int)
    arg_parser.add_argument("-dropout", default=0.33, type=float)
    # w2v_group = arg_parser.add_mutually_exclusive_group(required=True)
    arg_parser.add_argument("-w2v_size", default=300, type=int)
    arg_parser.add_argument("-pos_size", default=30, type=int)
    arg_parser.add_argument("-split_mlp_size", default=64, type=int)
    arg_parser.add_argument("-nuc_mlp_size", default=32, type=int)
    arg_parser.add_argument("-rel_mlp_size", default=128, type=int)
    arg_parser.add_argument("--w2v_freeze", dest="w2v_freeze", action="store_true")
    arg_parser.add_argument("-pretrained", default="data/pretrained/sgns.renmin.word")
    arg_parser.set_defaults(w2v_freeze=True)

    # train parameters
    arg_parser.add_argument("-epoch", default=20, type=int)
    arg_parser.add_argument("-batch_size", default=64, type=int)
    arg_parser.add_argument("-lr", default=0.001, type=float)
    arg_parser.add_argument("-l2", default=0.0, type=float)
    arg_parser.add_argument("-log_every", default=10, type=int)
    arg_parser.add_argument("-validate_every", default=10, type=int)
    arg_parser.add_argument("-a_split_loss", default=0.3, type=float)
    arg_parser.add_argument("-a_nuclear_loss", default=1.0, type=float)
    arg_parser.add_argument("-a_relation_loss", default=1.0, type=float)
    arg_parser.add_argument("-log_dir", default="data/log")
    arg_parser.add_argument("-model_save", default="data/models/treebuilder.partptr.model")
    arg_parser.add_argument("--seed", default=7, type=int)
    arg_parser.add_argument("--use_gpu", dest="use_gpu", action="store_true")
    arg_parser.set_defaults(use_gpu=True)

    main(arg_parser.parse_args())

