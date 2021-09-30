# coding: UTF-8
import argparse
import logging
import random
import torch
import copy
import numpy as np
from dataset import CDTB
from collections import Counter
from itertools import chain
from structure.vocab import Vocab, Label
from structure.nodes import node_type_filter, EDU, Relation, Sentence, TEXT
from treebuilder.dp_nr.model import PartitionPtr, Discriminator
from treebuilder.dp_nr.parser import PartitionPtrParser
import torch.optim as optim
from util.eval import parse_eval, parse_eval_all_info, gen_parse_report
from tensorboardX import SummaryWriter
from config import *
from torch.autograd import Variable as Var
from util.model_util import *


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


def gen_graph(splits):
    _, decoder_inputs, pred_splits, pred_nucs, pred_rels = splits
    len_ = decoder_inputs[0][1] - decoder_inputs[0][0] + 1
    split_num = len(decoder_inputs)
    graph_s, graph_nr, line_idx = [], [], [0]
    idx = 0
    queue_ = []
    pred_nr = []
    for idx_ in range(split_num):
        start, split, end, nuc, rel = decoder_inputs[idx_][0], pred_splits[idx_], decoder_inputs[idx_][1], \
                                      pred_nucs[idx_], pred_rels[idx_]
        pred_nr.append(nr2ids[str(nuc) + "-" + str(rel)])
        # Build figure
        if len(queue_) == 0:  # initialization
            queue_.append(end)
            graph_s.append([-2 for _ in range(len_)])
            graph_nr.append([0 for _ in range(len_)])
            graph_s[idx][split] = 0
            graph_nr[idx][split] = (pred_nr[-1] + 1)
        else:
            idx += 1
            if end > queue_[-1]:
                while end > queue_[-1]:
                    queue_.pop(-1)
                    idx -= 1
            if idx >= len(graph_s):
                graph_s.append([-2 for _ in range(len_)])
                graph_nr.append([0 for _ in range(len_)])
            queue_.append(end)
            graph_s[idx][split] = 0
            graph_nr[idx][split] = (pred_nr[-1] + 1)
            line_idx.append(idx)

        A = split - start == 1
        B = end - split == 1
        if A or B:
            if idx + 1 >= len(graph_s):
                graph_s.append([-2 for _ in range(len_)])
                graph_nr.append([0 for _ in range(len_)])
        if A:
            graph_s[idx + 1][split - 1] = -1
            graph_nr[idx + 1][split - 1] = -1
        if B:
            graph_s[idx + 1][split + 1] = -1
            graph_nr[idx + 1][split + 1] = -1

    return graph_s, graph_nr, line_idx


def gen_batch_iter(instances, batch_size, use_gpu=False):
    random_instances = np.random.permutation(instances)
    num_instances = len(instances)
    offset = 0
    while offset < num_instances:
        batch = random_instances[offset: min(num_instances, offset+batch_size)]

        # find out max seqlen of edus and words of edus
        num_batch = batch.shape[0]
        max_edu_seqlen = 0
        max_word_seqlen = 0
        for encoder_inputs, decoder_inputs, pred_splits, pred_nucs, pred_rels in batch:
            max_edu_seqlen = max_edu_seqlen if max_edu_seqlen >= len(encoder_inputs) else len(encoder_inputs)
            for edu_word_ids, edu_pos_ids in encoder_inputs:
                max_word_seqlen = max_word_seqlen if max_word_seqlen >= len(edu_word_ids) else len(edu_word_ids)

        # batch to numpy
        e_input_words = np.zeros([num_batch, max_edu_seqlen, max_word_seqlen], dtype=np.long)
        e_boundaries = np.zeros([num_batch, max_edu_seqlen + 1], dtype=np.long)
        e_input_poses = np.zeros([num_batch, max_edu_seqlen, max_word_seqlen], dtype=np.long)
        e_masks = np.zeros([num_batch, max_edu_seqlen, max_word_seqlen], dtype=np.uint8)

        d_inputs = np.zeros([num_batch, max_edu_seqlen-1, 2], dtype=np.long)
        d_outputs = np.zeros([num_batch, max_edu_seqlen-1], dtype=np.long)
        d_output_nr = np.zeros([num_batch, max_edu_seqlen - 1], dtype=np.long)
        d_masks = np.zeros([num_batch, max_edu_seqlen-1, max_edu_seqlen+1], dtype=np.uint8)

        # GAN use
        d_g_s = np.zeros([num_batch, MAX_H, MAX_W], dtype=np.float)
        d_g_nr = np.zeros([num_batch, MAX_H, MAX_W], dtype=np.float)
        l_x = list()

        for batchi, (encoder_inputs, decoder_inputs, pred_splits, pred_nucs, pred_rels) in enumerate(batch):
            for edui, (edu_word_ids, edu_pos_ids) in enumerate(encoder_inputs):
                # input(edu_word_ids)
                if edu_word_ids[-1] in [39, 3015, 178]:
                    e_boundaries[batchi][edui + 1] = 1  # 句子边界
                word_seqlen = len(edu_word_ids)
                e_input_words[batchi][edui][:word_seqlen] = edu_word_ids
                e_input_poses[batchi][edui][:word_seqlen] = edu_pos_ids
                e_masks[batchi][edui][:word_seqlen] = 1
            e_boundaries[batchi][-1] = 0
            for di, decoder_input in enumerate(decoder_inputs):
                d_inputs[batchi][di] = decoder_input
                d_masks[batchi][di][decoder_input[0]+1: decoder_input[1]] = 1
            d_outputs[batchi][:len(pred_splits)] = pred_splits
            nrs = [nr2ids[str(n) + "-" + str(r)] for n, r in zip(pred_nucs, pred_rels)]
            d_output_nr[batchi][:len(nrs)] = nrs

            # graph generation
            g_s, g_nr, x = gen_graph((encoder_inputs, decoder_inputs, pred_splits, pred_nucs, pred_rels))

            h_, w_ = len(g_s), len(g_s[0])
            w_ = min(w_, MAX_W)
            h_ = min(h_, MAX_H)
            for idx in range(h_):
                d_g_s[batchi][idx][:w_] = g_s[idx][:w_]
                d_g_nr[batchi][idx][:w_] = g_nr[idx][:w_]
            l_x.append(x)

        # numpy to torch
        e_input_words = torch.from_numpy(e_input_words).long()
        e_boundaries = torch.from_numpy(e_boundaries).long()
        e_input_poses = torch.from_numpy(e_input_poses).long()
        e_masks = torch.from_numpy(e_masks).byte()
        d_inputs = torch.from_numpy(d_inputs).long()
        d_outputs = torch.from_numpy(d_outputs).long()
        d_output_nr = torch.from_numpy(d_output_nr).long()
        d_masks = torch.from_numpy(d_masks).byte()
        d_g_s = torch.from_numpy(d_g_s).float()
        d_g_nr = torch.from_numpy(d_g_nr).float()

        if use_gpu:
            e_input_words = e_input_words.cuda()
            e_boundaries = e_boundaries.cuda()
            e_input_poses = e_input_poses.cuda()
            e_masks = e_masks.cuda()
            d_inputs = d_inputs.cuda()
            d_outputs = d_outputs.cuda()
            d_output_nr = d_output_nr.cuda()
            d_masks = d_masks.cuda()
            d_g_s = d_g_s.cuda()
            d_g_nr = d_g_nr.cuda()
        yield (e_input_words, e_input_poses, e_masks, e_boundaries), (d_inputs, d_masks), \
              (d_outputs, d_output_nr, d_g_s, d_g_nr, l_x)
        offset = offset + batch_size


def parse_and_eval(dataset_val, model):
    model.eval()
    parser = PartitionPtrParser(model)
    golds = list(filter(lambda d: d.root_relation(), chain(*dataset_val)))
    num_instances = len(golds)
    strips = []
    for paragraph in golds:
        edus = []
        for edu in paragraph.edus():
            edu_copy = EDU([TEXT(edu.text)])
            setattr(edu_copy, "words", edu.words)
            setattr(edu_copy, "tags", edu.tags)
            edus.append(edu_copy)
        strips.append(edus)
    parses = []
    for edus in strips:
        parse = parser.parse(edus)
        parses.append(parse)
    return num_instances, parse_eval_all_info(parses, golds)


def model_score(scores):
    eval_score = sum(score[2] for score in scores)
    return eval_score


def main(args):
    # set seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load dataset
    cdtb = CDTB(args.data, "TRAIN", "VALIDATE", "TEST", ctb_dir=args.ctb_dir, preprocess=True, cache_dir=args.cache_dir)
    # build vocabulary
    word_vocab, pos_vocab, nuc_label, rel_label = build_vocab(cdtb.train)
    trainset = numericalize(cdtb.train, word_vocab, pos_vocab, nuc_label, rel_label)
    logging.info("num of instances trainset: %d" % len(trainset))
    logging.info("args: %s" % str(args))
    # build model
    model = PartitionPtr(hidden_size=args.hidden_size, dropout=args.dropout,
                         word_vocab=word_vocab, pos_vocab=pos_vocab, nuc_label=nuc_label, rel_label=rel_label,
                         pretrained=args.pretrained, w2v_size=args.w2v_size, w2v_freeze=args.w2v_freeze,
                         pos_size=args.pos_size,
                         split_mlp_size=args.split_mlp_size, nuc_mlp_size=args.nuc_mlp_size,
                         rel_mlp_size=args.rel_mlp_size,
                         use_gpu=args.use_gpu)
    discriminator = Discriminator()
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))
    if args.use_gpu:
        model.cuda()
        discriminator.cuda()
    logging.info("model:\n%s" % str(model))

    # train and evaluate
    niter = 0
    log_splits_loss = 0.
    log_nr_loss = 0.
    # log_loss = 0.
    gen_loss, dis_loss = 0, 0.
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    best_model_score = 0.
    best_performance = 0.
    report_results = [0., 0., 0., 0.]
    final_info = None
    for nepoch in range(1, args.epoch + 1):
        batch_iter = gen_batch_iter(trainset, args.batch_size, args.use_gpu)
        for nbatch, (e_inputs, d_inputs, grounds) in enumerate(batch_iter, start=1):
            niter += 1
            model.train()
            optimizer.zero_grad()
            splits_loss, nr_loss, gen_img, real_img = model.loss(e_inputs, d_inputs, grounds, nepoch)
            loss = args.a_split_loss * splits_loss + args.a_relation_loss * nr_loss
            if nepoch > WARM_UP_EP:
                # GAN
                gen_feat = model.cnn_feat_ext(gen_img)
                gen_label = discriminator(gen_feat)
                g_loss = 0.5 * torch.mean((gen_label - 1)**2)
            else:
                g_loss = 0.
            loss_gen = loss + g_loss
            loss_gen.backward()
            optimizer.step()
            log_splits_loss += splits_loss.item()
            log_nr_loss += nr_loss.item()

            # Discriminator
            if nepoch > WARM_UP_EP:
                # discriminator
                # (d_g_s, d_g_n, d_g_r), (gen_g_s, gen_g_n, gen_g_r)
                optimizer_D.zero_grad()
                # gen img feat extraction
                real_feat = discriminator.cnn_feat_ext(real_img)
                gen_feat = model.cnn_feat_ext(gen_img.detach())
                # Measure discriminator's ability to classify real from generated samples
                d_real = discriminator(real_feat)
                d_fake = discriminator(gen_feat)

                if LABEL_SWITCH and niter % SWITCH_ITE == 0:
                    d_loss = 0.5 * (torch.mean((d_fake - 1) ** 2) + torch.mean(d_real ** 2))
                else:
                    d_loss = 0.5 * (torch.mean((d_real - 1) ** 2) + torch.mean(d_fake ** 2))
                d_loss.backward()
                optimizer_D.step()
                dis_loss += d_loss.item()
                gen_loss += g_loss.item()

            if niter % args.log_every == 0:
                print_("=============================================", args.log_dir)
                print_("-iter " + str(niter) + " -epoch " + str(nepoch) + " -batch " + str(nbatch) +
                       " \n[s_loss: " + str(log_splits_loss) + ", nr_loss: " + str(log_nr_loss) +
                       ", \ngen_loss: " + str(gen_loss) + ", dis_loss: " + str(dis_loss) + "]", args.log_dir)
                log_splits_loss = 0.
                log_nr_loss = 0.
                gen_loss = 0.
                dis_loss = 0.
            # span_score, nuc_score, ctype_score, ftype_score, cfull_score, ffull_score
            if niter % args.validate_every == 0:
                num_instances, validate_scores_ = parse_and_eval(cdtb.validate, model)
                validate_scores, _ = validate_scores_
                print_("Validate on " + str(num_instances) + " instances", args.log_dir)
                print_("[Span: " + str(round(validate_scores[0][2], 4)) +
                       " Nucl: " + str(round(validate_scores[1][2], 4)) +
                       " Rel: " + str(round(validate_scores[2][2], 4)) +
                       " Full: " + str(round(validate_scores[3][2], 4)), args.log_dir)
                if model_score(validate_scores) > best_model_score:
                    # test on test set with new best model
                    best_model_score = model_score(validate_scores)
                    best_model = copy.deepcopy(model)
                    num_instances, test_scores_ = parse_and_eval(cdtb.test, best_model)
                    test_scores, info_ = test_scores_
                    if model_score(test_scores) > best_performance:
                        final_info = info_
                        best_performance = model_score(test_scores)
                        report_results = [item[2] for item in test_scores]
                        with open(args.model_save, "wb+") as model_fd:
                            torch.save(best_model, model_fd)
                    print_("-- Dev better & Test on " + str(num_instances) + " instances --", args.log_dir)
                    print_("[Span: " + str(round(report_results[0], 4)) +
                           " Nucl: " + str(round(report_results[1], 4)) +
                           " Rel: " + str(round(report_results[2], 4)) +
                           " Full: " + str(round(report_results[3], 4)), args.log_dir)
                    break
    print_("", args.log_dir)
    print_("=================== Final ===================", args.log_dir)
    print_("[Span: " + str(round(report_results[0], 4)) +
           " Nuclearity: " + str(round(report_results[1], 4)) +
           " Relation: " + str(round(report_results[2], 4)) +
           " Full: " + str(round(report_results[3], 4)), args.log_dir)
    print_(obtain_rel_info(final_info), args.log_dir)
    print_("=================== End ===================", args.log_dir)


def obtain_rel_info(final_info):
    p_list, r_list, f_list = dict(), dict(), dict()
    if final_info is None:
        return "None"
    else:
        rel_pre_true, rel_gold, rel_pre_all = final_info
        rels = rel_gold.keys()
        for rel_idx in rels:
            c_ = rel_pre_true[rel_idx] if rel_idx in rel_pre_true.keys() else 0
            g_ = rel_gold[rel_idx] if rel_idx in rel_gold.keys() else 0
            h_ = rel_pre_all[rel_idx] if rel_idx in rel_pre_all.keys() else 0
            p_b = 0. if h_ == 0 else c_ / h_
            r_b = 0. if g_ == 0 else c_ / g_
            f_b = 0. if (g_ + h_) == 0 else (2 * c_) / (g_ + h_)
            p_list[rel_idx] = p_b
            r_list[rel_idx] = r_b
            f_list[rel_idx] = f_b
        rels = " ".join(rels)
        p_info = " ".join([str(p_) for p_ in p_list.values()])
        r_info = " ".join([str(p_) for p_ in r_list.values()])
        f_info = " ".join([str(p_) for p_ in f_list.values()])
        return rels + "\n" + p_info + "\n" + r_info + "\n" + f_info


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_parser = argparse.ArgumentParser()

    # dataset parameters
    arg_parser.add_argument("--data", default="data/CDTB")
    arg_parser.add_argument("--ctb_dir", default="data/CTB")
    arg_parser.add_argument("--cache_dir", default=None)
    arg_parser.add_argument("-set", default="Null")

    # model parameters
    arg_parser.add_argument("-hidden_size", default=512, type=int)
    arg_parser.add_argument("-dropout", default=0.33, type=float)
    # w2v_group = arg_parser.add_mutually_exclusive_group(required=True)
    arg_parser.add_argument("-w2v_size", default=300, type=int)
    arg_parser.add_argument("-pos_size", default=30, type=int)
    arg_parser.add_argument("-split_mlp_size", default=64, type=int)
    arg_parser.add_argument("-nuc_mlp_size", default=32, type=int)  # 32
    arg_parser.add_argument("-rel_mlp_size", default=128, type=int)
    arg_parser.add_argument("--w2v_freeze", dest="w2v_freeze", action="store_true")
    arg_parser.add_argument("-pretrained", default="data/pretrained/sgns.renmin.word")
    arg_parser.set_defaults(w2v_freeze=True)

    # train parameters
    arg_parser.add_argument("-epoch", default=20, type=int)
    arg_parser.add_argument("-batch_size", default=64, type=int)
    arg_parser.add_argument("-lr", default=0.001, type=float)
    arg_parser.add_argument("-l2", default=5e-5, type=float)
    arg_parser.add_argument("-log_every", default=5, type=int)
    arg_parser.add_argument("-validate_every", default=10, type=int)
    arg_parser.add_argument("-a_split_loss", default=0.1, type=float)
    arg_parser.add_argument("-a_nuclear_loss", default=1.0, type=float)
    arg_parser.add_argument("-a_relation_loss", default=1.0, type=float)
    arg_parser.add_argument("-log_dir", default="data/log/set_" + str(SET) + ".tsv")
    arg_parser.add_argument("-model_save", default="data/models/treebuilder.partptr.model" + str(SET))
    arg_parser.add_argument("--seed", default=SEED, type=int)
    arg_parser.add_argument("--use_gpu", dest="use_gpu", action="store_true")
    arg_parser.set_defaults(use_gpu=True)

    main(arg_parser.parse_args())
