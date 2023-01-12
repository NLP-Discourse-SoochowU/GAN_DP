# coding: utf-8
from structure import EDU, Sentence, Relation
import numpy as np


def factorize_tree(tree, binarize=True):
    quads = set()   # (span, nuclear, coarse relation, fine relation)

    def factorize(root, offset=0):
        if isinstance(root, EDU):
            return [(offset, offset+len(root.text))]
        elif isinstance(root, Sentence):
            children_spans = []
            for child in root:
                spans = factorize(child, offset)
                children_spans.extend(spans)
                offset = spans[-1][1]
            return children_spans
        elif isinstance(root, Relation):
            children_spans = []
            for child in root:
                spans = factorize(child, offset)
                children_spans.extend(spans)
                offset = spans[-1][1]
            if binarize:
                while len(children_spans) >= 2:
                    right = children_spans.pop()
                    left = children_spans.pop()
                    quads.add(((left, right), root.nuclear, root.ctype, root.ftype))
                    children_spans.append((left[0], right[1]))
            else:
                quads.add((tuple(children_spans), root.nuclear, root.ctype, root.ftype))
            return [(children_spans[0][0], children_spans[-1][1])]

    factorize(tree.root_relation())
    return quads


def f1_score(num_corr, num_gold, num_pred, treewise_average=True):
    num_corr = num_corr.astype(np.float)
    num_gold = num_gold.astype(np.float)
    num_pred = num_pred.astype(np.float)

    if treewise_average:
        precision = (np.nan_to_num(num_corr / num_pred)).mean()
        recall = (np.nan_to_num(num_corr / num_gold)).mean()
    else:
        precision = np.nan_to_num(num_corr.sum() / num_pred.sum())
        recall = np.nan_to_num(num_corr.sum() / num_gold.sum())

    if precision + recall == 0:
        f1 = 0.
    else:
        f1 = 2. * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluation_trees(parses, golds, binarize=True, treewise_avearge=True):
    num_gold = np.zeros(len(golds))
    num_parse = np.zeros(len(parses))
    num_corr_span = np.zeros(len(parses))
    num_corr_nuc = np.zeros(len(parses))
    num_corr_ctype = np.zeros(len(parses))
    num_corr_ftype = np.zeros(len(parses))

    for i, (parse, gold) in enumerate(zip(parses, golds)):
        parse_factorized = factorize_tree(parse, binarize=binarize)
        gold_factorized = factorize_tree(gold, binarize=binarize)
        num_parse[i] = len(parse_factorized)
        num_gold[i] = len(gold_factorized)

        # index quads by child spans
        parse_dict = {quad[0]: quad for quad in parse_factorized}
        gold_dict = {quad[0]: quad for quad in gold_factorized}

        # find correct spans
        gold_spans = set(gold_dict.keys())
        parse_spans = set(parse_dict.keys())
        corr_spans = gold_spans & parse_spans
        num_corr_span[i] = len(corr_spans)

        # quad [span, nuclear, ftype, ctype]
        for span in corr_spans:
            # count correct nuclear
            num_corr_nuc[i] += 1 if parse_dict[span][1] == gold_dict[span][1] else 0
            # count correct ctype
            num_corr_ctype[i] += 1 if parse_dict[span][2] == gold_dict[span][2] else 0
            # count correct ftype
            num_corr_ftype[i] += 1 if parse_dict[span][3] == gold_dict[span][3] else 0

    span_score = f1_score(num_corr_span, num_gold, num_parse, treewise_average=treewise_avearge)
    nuc_score = f1_score(num_corr_nuc, num_gold, num_parse, treewise_average=treewise_avearge)
    ctype_score = f1_score(num_corr_ctype, num_gold, num_parse, treewise_average=treewise_avearge)
    ftype_score = f1_score(num_corr_ftype, num_gold, num_parse, treewise_average=treewise_avearge)
    return span_score, nuc_score, ctype_score, ftype_score
