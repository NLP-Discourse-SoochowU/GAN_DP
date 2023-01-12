# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date: 2019.1.24
@Description: The trainer of our parser.
"""
import logging
import random
import numpy as np
import progressbar
import torch.optim as optim
from util.radam import AdamW
from util.DL_tricks import *
from config import *
from model.stacked_parser_tdt_gan3.model import PartitionPtr, Discriminator
from model.metric import Metrics
from model.stacked_parser_tdt_gan3.parser import PartitionPtrParser
from structure.rst import RST
from structure.tree_obj import tree_obj

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


class Trainer:
    def __init__(self):
        self.metric = Metrics()
        rst = RST()
        word2ids, pos2ids = rst.get_voc_labels()
        self.train_set = rst.train
        self.dev_set = rst.tree_obj_dev
        self.test_set = rst.tree_obj_test
        self.model = PartitionPtr(word2ids=word2ids, pos2ids=pos2ids, nuc2ids=nucl2ids, rel2ids=coarse2ids,
                                  pretrained=load_data(VOC_VEC_PATH))
        self.discriminator = Discriminator()
        self.discriminator.apply(weights_init_normal)
        self.loss_all = []
        if USE_CUDA:
            self.model.cuda(CUDA_ID)
            self.discriminator.cuda(CUDA_ID)
        logging.info("basic:\n%s" % str(self.model))

    def train(self, test_desc):
        """ The main process of training and evaluating.
        """
        log_file = os.path.join(LOG_ALL, "set_" + str(SET) + ".log")
        n_iter, log_s_loss, log_nr_loss, log_loss, gen_loss, dis_loss = 0, 0., 0., 0., 0., 0.
        optimizer = optim.Adam(self.model.parameters(), lr=LEARN_RATE, weight_decay=L2) if not USE_R_ADAM else \
            AdamW(self.model.parameters(), lr=LEARN_RATE, betas=(BETA1, BETA2), weight_decay=WD, warmup=WARM_UP)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=D_LR, betas=(0.5, 0.999))
        p = progressbar.ProgressBar()
        k, gamma, lambda_k, M = 0., 0.75, 0.001, 0.
        for n_epoch in range(1, EPOCH + 1):
            p.start(LOG_EVERY)
            batch_iter = self.gen_batch_iter()
            for n_batch, (e_inputs, d_inputs, gold_labels) in enumerate(batch_iter, start=1):
                self.model.train()
                if e_inputs[0].size(0) < BATCH_SIZE:
                    break
                n_iter += 1
                p.update((n_iter % LOG_EVERY))
                splits_loss, nr_loss, gen_img, real_img = self.model.loss(e_inputs, d_inputs, gold_labels, n_epoch)
                loss = ALPHA_SPAN * splits_loss + ALPHA_NR * nr_loss
                if n_epoch > WARM_UP_EP:
                    # GAN
                    gen_feat = self.model.cnn_feat_ext(gen_img)
                    gen_label = self.discriminator(gen_feat)
                    g_loss = 0.5 * torch.mean((gen_label - 1) ** 2)
                else:
                    g_loss = 0.
                loss_gen = loss + g_loss
                optimizer.zero_grad()
                loss_gen.backward()
                optimizer.step()
                if n_epoch > WARM_UP_EP:
                    # discriminator
                    # (d_g_s, d_g_n, d_g_r), (gen_g_s, gen_g_n, gen_g_r)
                    optimizer_D.zero_grad()
                    # gen img feat extraction
                    real_feat = self.discriminator.cnn_feat_ext(real_img)
                    gen_feat = self.model.cnn_feat_ext(gen_img.detach())
                    # Measure discriminator's ability to classify real from generated samples
                    d_real = self.discriminator(real_feat)
                    d_fake = self.discriminator(gen_feat)

                    if LABEL_SWITCH and n_iter % SWITCH_ITE == 0:
                        d_loss = 0.5 * (torch.mean((d_fake - 1) ** 2) + torch.mean(d_real ** 2))
                    else:
                        d_loss = 0.5 * (torch.mean((d_real - 1) ** 2) + torch.mean(d_fake ** 2))
                    d_loss.backward()
                    optimizer_D.step()
                    dis_loss += d_loss.item()
                    gen_loss += g_loss.item()
                log_s_loss += splits_loss.item()
                log_nr_loss += nr_loss.item()

                if n_iter % LOG_EVERY == 0:
                    p.finish()
                    logging.info("iter-%-2d, epoch-%-2d, batch-%-2d (Span==>%.2f, NR==>%.2f, G==>%.5f, D==>%.5f, "
                                 "M==>%.2f)" %
                                 (n_iter, n_epoch, n_batch, log_s_loss, log_nr_loss, gen_loss, dis_loss, M))
                    self.loss_all.append((log_s_loss, log_nr_loss, gen_loss, dis_loss, M))
                    log_s_loss, log_nr_loss, gen_loss, dis_loss = 0., 0., 0., 0.
                    p.start(LOG_EVERY)
                if n_iter % VALIDATE_EVERY == 0:
                    if self.parse_and_eval(RST.get_dev(), save_per=True, type_="dev"):
                        self.parse_and_eval(RST.get_test(), save_per=False, type_="test")
                    self.report(n_epoch, n_iter, test_desc, log_file)
        save_data(self.loss_all, LOSS_PATH + "SET_" + str(SET) + ".pkl")

    def gen_batch_iter(self):
        """ instances:
            (encoder_inputs, decoder_inputs, pred_splits, pred_nucl, pred_rels)
            encoder_inputs: (edu_word_ids, edu_pos_ids)
            decoder_inputs: (decoder_input))
        """
        instances = self.train_set
        random_instances = np.random.permutation(instances)
        num_instances = len(instances)
        offset = 0
        while offset < num_instances:
            batch = random_instances[offset: min(num_instances, offset + BATCH_SIZE)]
            num_batch = batch.shape[0]
            max_edu_len = 0
            max_word_len = 0
            for encoder_inputs, _, _, _, g_s, _, _, _, _ in batch:
                max_edu_len = max_edu_len if max_edu_len >= len(encoder_inputs) else len(encoder_inputs)
                for edu_word_ids, _, _, _ in encoder_inputs:
                    max_word_len = max_word_len if max_word_len >= len(edu_word_ids) else len(edu_word_ids)

            # batch2numpy encode
            e_in_words = np.zeros([num_batch, max_edu_len, max_word_len], dtype=np.long)
            e_in_word_embeds = np.zeros([num_batch, max_edu_len, max_word_len, 1024], dtype=np.long)
            e_in_edu_bound = np.zeros([num_batch, max_edu_len - 1], dtype=np.long)
            e_in_poses = np.zeros([num_batch, max_edu_len, max_word_len], dtype=np.long)
            e_masks = np.zeros([num_batch, max_edu_len, max_word_len], dtype=np.uint8)

            # batch2numpy decode
            d_inputs = np.zeros([num_batch, max_edu_len - 1, 2], dtype=np.long)
            d_outputs = np.zeros([num_batch, max_edu_len - 1], dtype=np.long)
            d_output_nr = np.zeros([num_batch, max_edu_len - 1], dtype=np.long)
            d_masks = np.zeros([num_batch, max_edu_len - 1, max_edu_len - 1], dtype=np.uint8)
            d_g_s = -1 * np.ones([num_batch, MAX_H, MAX_W], dtype=np.float)
            d_g_nr = np.zeros([num_batch, MAX_H, MAX_W], dtype=np.float)
            l_x = list()

            for batch_idx, (encoder_inputs, decoder_inputs, pred_splits, pred_nr, g_s, _, _, g_nr, x) in enumerate(batch):
                for edu_idx, (edu_word_ids, edu_word_embed, edu_pos_ids, bound_info) in enumerate(encoder_inputs):
                    word_len = len(edu_word_ids)
                    e_in_words[batch_idx][edu_idx][:word_len] = edu_word_ids
                    if edu_idx < max_edu_len - 1:
                        e_in_edu_bound[batch_idx][edu_idx] = bound_info
                    e_in_poses[batch_idx][edu_idx][:word_len] = edu_pos_ids
                    e_masks[batch_idx][edu_idx][:word_len] = 1
                    for idx_ in range(word_len):
                        e_in_word_embeds[batch_idx][edu_idx][idx_][:] = edu_word_embed[idx_].detach().numpy()
                for d_idx, decoder_input in enumerate(decoder_inputs):
                    d_inputs[batch_idx][d_idx] = (decoder_input[0], decoder_input[1] - 2)
                    d_masks[batch_idx][d_idx][decoder_input[0]: decoder_input[1] - 1] = 1
                d_outputs[batch_idx][:len(pred_splits)] = [item - 1 for item in pred_splits]
                d_output_nr[batch_idx][:len(pred_nr)] = pred_nr
                l_x.append(x)
                h_, w_ = len(g_s), len(g_s[0])
                w_ = min(w_, MAX_W)
                h_ = min(h_, MAX_H)
                for idx in range(h_):
                    d_g_s[batch_idx][idx][:w_] = g_s[idx][:w_]
                    d_g_nr[batch_idx][idx][:w_] = g_nr[idx][:w_]

            # numpy2torch
            e_in_words = torch.from_numpy(e_in_words).long()
            e_in_edu_bound = torch.from_numpy(e_in_edu_bound).long()
            e_in_word_embeds = torch.from_numpy(e_in_word_embeds).float()
            e_in_poses = torch.from_numpy(e_in_poses).long()
            e_masks = torch.from_numpy(e_masks).byte()
            d_inputs = torch.from_numpy(d_inputs).long()
            d_outputs = torch.from_numpy(d_outputs).long()
            d_output_nr = torch.from_numpy(d_output_nr).long()
            d_masks = torch.from_numpy(d_masks).byte()
            d_g_s = torch.from_numpy(d_g_s).float()
            d_g_nr = torch.from_numpy(d_g_nr).float()

            if USE_CUDA:
                if not USE_ELMo:
                    e_in_words = e_in_words.cuda(CUDA_ID)
                e_in_poses = e_in_poses.cuda(CUDA_ID)
                e_masks = e_masks.cuda(CUDA_ID)
                e_in_word_embeds = e_in_word_embeds.cuda(CUDA_ID)
                d_inputs = d_inputs.cuda(CUDA_ID)
                d_outputs = d_outputs.cuda(CUDA_ID)
                d_output_nr = d_output_nr.cuda(CUDA_ID)
                d_masks = d_masks.cuda(CUDA_ID)
                e_in_edu_bound = e_in_edu_bound.cuda(CUDA_ID)
                d_g_s = d_g_s.cuda(CUDA_ID)
                d_g_nr = d_g_nr.cuda(CUDA_ID)

            yield (e_in_words, e_in_word_embeds, e_in_poses, e_masks, e_in_edu_bound), (d_inputs, d_masks), \
                  (d_outputs, d_output_nr, d_g_s, d_g_nr, l_x)
            offset = offset + BATCH_SIZE

    def parse_and_eval(self, trees2parse, save_per, type_):
        gold_trees = self.dev_set if type_ == "dev" else self.test_set
        self.model.eval()
        trees_eval_pred = []
        parser = PartitionPtrParser(self.model)
        for tree in trees2parse:
            parsed_tree = parser.parse(tree.edus)
            trees_eval_pred.append(tree_obj(parsed_tree))
        better = self.metric.eval_(gold_trees, trees_eval_pred, self.model, type_=type_)
        return better

    def report(self, epoch, iter_, test_desc, log_file):
        message = self.metric.get_scores()
        info_str = "Epoch: " + str(epoch) + "   iteration: " + str(iter_) + "  desc: " + test_desc
        logging.info(info_str)
        print_(str_=info_str, log_file=log_file)
        for msg in message:
            logging.info(msg)
            print_(str_=msg, log_file=log_file)
