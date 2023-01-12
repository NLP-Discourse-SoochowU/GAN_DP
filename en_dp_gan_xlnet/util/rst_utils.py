# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date: 2018.4.5
@Description:
"""
from config import *
from path_config import *
from util.file_util import *
from allennlp.modules.elmo import batch_to_ids
from allennlp.modules.elmo import Elmo
from util.patterns import upper_re

elmo = Elmo(options_file, weight_file, 2, dropout=0)
word2ids = load_data(VOC_WORD2IDS_PATH)
pos2ids = load_data(POS_word2ids_PATH)
print("Load Done.")


def get_one_edu_info(ori_edu, edu_txt, nlp=None):
    if upper_re.match(ori_edu) is not None:
        edu_txt = edu_txt[0].lower() + edu_txt[1:]
    toks, tags = tok_analyse(nlp, edu_txt)
    edu_ids = []
    for tok in toks:
        if tok in word2ids.keys():
            edu_ids.append(word2ids[tok])
        else:
            edu_ids.append(word2ids[UNK])
    pos_ids = [pos2ids[tag] for tag in tags]
    edu_emlo = None
    toks_ids = batch_to_ids([toks])  # (1, sent_len)
    toks_emb = elmo(toks_ids)["elmo_representations"][0][0]
    for tmp_token_emb in toks_emb:
        tmp_token_emb = tmp_token_emb.unsqueeze(0)
        edu_emlo = tmp_token_emb if edu_emlo is None else torch.cat((edu_emlo, tmp_token_emb), 0)
    return edu_txt, edu_ids, pos_ids, edu_emlo


def get_edus_info(edu_path, sentence_path, nlp=None):
    """ Difficult, complex, once for all.
    """
    sent_list, sent_list4bound = [], []
    with open(sentence_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            sent_list.append(line)
            sent_list4bound.append(line)

    e_li = []
    e_bound_li = []
    tmp_e_concat = ""
    tmp_s_concat = "".join(sent_list4bound.pop(0).split())
    with open(edu_path, "r") as f:
        for line in f:
            temp_e = line.strip()
            e_li.append(temp_e)
            tmp_e_concat += ("".join(temp_e.split()))
            if tmp_e_concat == tmp_s_concat:
                e_bound_li.append(True)
                # initialize
                if len(sent_list4bound) > 0:
                    tmp_s_concat = "".join(sent_list4bound.pop(0).split())
                    tmp_e_concat = ""

            elif len(tmp_e_concat) > len(tmp_s_concat):
                while len(tmp_e_concat) > len(tmp_s_concat):
                    tmp_s_concat += "".join(sent_list4bound.pop(0).split())

                if tmp_e_concat == tmp_s_concat:
                    e_bound_li.append(True)
                    # initialize
                    if len(sent_list4bound) > 0:
                        tmp_s_concat = "".join(sent_list4bound.pop(0).split())
                        tmp_e_concat = ""
                else:
                    e_bound_li.append(False)
            else:
                e_bound_li.append(False)

    e_token_list, e_token_ids_li, e_emlo_li, s_token_list = convert2tokens_ids(e_li, sent_list, nlp)
    e_tags_li, e_span_li, e_headwords_li, e_cent_words_li = [], [], [], []
    tag_ids_list, dep_ids_list = [], []
    for s_tokens in s_token_list:
        sent_text = " ".join(s_tokens)
        tags = tok_analyse(nlp, sent_text)[1]
        for tag in tags:
            tmp_tag_id = pos2ids[tag] if tag in pos2ids.keys() else pos2ids[UNK]
            tag_ids_list.append(tmp_tag_id)

        dep_tuple_list = nlp.dependency_parse(sent_text)
        if len(s_tokens) != len(dep_tuple_list):
            dependency_more = len(dep_tuple_list) - len(s_tokens)
            if dependency_more > 0:
                dep_tuple_list = dep_tuple_list[:-dependency_more]

        tmp_dep_dict, count_tuple = dict(), 0
        for tuple_ in dep_tuple_list:
            if tuple_[1] == 0 and count_tuple > 0:
                for idx in range(1, count_tuple + 1):
                    dep_ids_list.append(tmp_dep_dict[idx])
                tmp_dep_dict, count_tuple = dict(), 0
                word_ids = None
            else:
                word = s_tokens[tuple_[1]-1]
                word_ids = word2ids[word] if word in word2ids.keys() else word2ids[UNK]
            tmp_dep_dict[tuple_[2]] = word_ids
            count_tuple += 1
        for idx in range(1, count_tuple + 1):
            dep_ids_list.append(tmp_dep_dict[idx])

    offset = 0
    for one_edu_tokens in e_token_list:
        e_span_li.append((offset, offset + len(one_edu_tokens) - 1))
        e_cent_words_li.append(False)
        temp_tags_ids, temp_dep_idx = [], []
        for _ in one_edu_tokens:
            temp_tags_ids.append(tag_ids_list.pop(0))
            dep_word_idx = dep_ids_list.pop(0)
            if dep_word_idx is None:
                e_cent_words_li[-1] = True
                temp_dep_idx.append(PAD_ids)
            else:
                temp_dep_idx.append(dep_word_idx)
        e_tags_li.append(temp_tags_ids)
        e_headwords_li.append(temp_dep_idx)
        offset += len(one_edu_tokens)
    return e_bound_li, e_li, e_span_li, e_token_ids_li, e_tags_li, e_headwords_li, e_cent_words_li, e_emlo_li


def convert2tokens_ids(e_li, s_li, nlp):
    sents_tok_li = []
    edus_tok_li, edus_tok_ids_li = [], []
    edus_emlo_li = []  # ELMo

    tmp_s_ids = 0
    tmp_sent_ = s_li[0]

    tmp_e_tok_li = []
    for edu in e_li:
        tmp_e_toks = tok_analyse(nlp, edu)[0]
        tmp_e_tok_li.append(tmp_e_toks)
        judge_out = judge_equal(tmp_sent_, tmp_e_tok_li)
        if judge_out == 0:
            continue
        elif judge_out == 2:
            while judge_out == 2:
                tmp_s_ids += 1
                tmp_sent_ = tmp_sent_ + " " + s_li[tmp_s_ids]
                judge_out = judge_equal(tmp_sent_, tmp_e_tok_li)

        if judge_out == 1:
            flag = True
            sent_toks_li = None
            edus_len_li = []
            for tmp_e_toks in tmp_e_tok_li:
                tmp_tok_ids = []
                for idx in range(len(tmp_e_toks)):
                    if tmp_e_toks[idx] == "\"":
                        tmp_e_toks[idx] = "``" if flag else "''"
                        flag = not flag
                    tok = tmp_e_toks[idx].lower()
                    tok_ids = word2ids[tok] if tok in word2ids.keys() else word2ids[UNK]
                    tmp_tok_ids.append(tok_ids)
                edus_tok_li.append(tmp_e_toks[:])
                edus_tok_ids_li.append(tmp_tok_ids[:])
                if sent_toks_li is None:
                    sent_toks_li = tmp_e_toks
                else:
                    sent_toks_li.extend(tmp_e_toks)
                edus_len_li.append(len(tmp_e_toks))
            sents_tok_li.append(sent_toks_li)
            # ELMo
            edus_emlo_li += get_elmo_emb(edus_len_li, sent_toks_li)
            # init
            tmp_e_tok_li = []
            tmp_s_ids += 1
            tmp_sent_ = s_li[tmp_s_ids] if tmp_s_ids < len(s_li) else None

    return edus_tok_li, edus_tok_ids_li, edus_emlo_li, sents_tok_li


def get_elmo_emb(e_len_li, s_tok_li):
    """ (edu_num, word_num, id_len)
    """
    e_emb_li = []
    sents_toks_ids = batch_to_ids([s_tok_li])  # (1, sent_len)
    tmp_s_toks_emb = elmo(sents_toks_ids)["elmo_representations"][0][0]
    tmp_e_emb = None
    tok_idx = 0
    for len_ in e_len_li:
        for _ in range(len_):
            tmp_token_emb = tmp_s_toks_emb[tok_idx].unsqueeze(0)
            tmp_e_emb = tmp_token_emb if tmp_e_emb is None else torch.cat((tmp_e_emb, tmp_token_emb), 0)
            tok_idx += 1
        e_emb_li.append(tmp_e_emb)
        tmp_e_emb = None
    return e_emb_li


def judge_equal(sent_raw, edus_token_list):
    flag = 0
    sent_raw = "".join(sent_raw.split())
    sents_edu = []
    for edus_tokens in edus_token_list:
        edu_tokens = []
        for token in edus_tokens:
            token_ = "".join(token.split())
            edu_tokens.append(token_)
        sents_edu.append(edu_tokens[:])
    sent_temp = "".join(["".join(edu_tokens) for edu_tokens in sents_edu])
    sent_raw = sent_raw.lower()
    sent_temp = sent_temp.lower()
    if sent_raw == sent_temp:
        flag = 1
    if len(sent_raw) < len(sent_temp):
        flag = 2
    return flag


def tok_analyse(nlp, edu):
    tok_pairs = nlp.pos_tag(edu)
    words = [pair[0] for pair in tok_pairs]
    tags = [pair[1] for pair in tok_pairs]
    return words, tags
