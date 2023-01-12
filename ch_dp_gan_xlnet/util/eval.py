# coding: UTF-8
from collections import defaultdict, Counter
from itertools import chain
import numpy as np
from structure.nodes import EDU, Sentence, Relation


def edu_eval(segs, golds):
    num_corr = 0
    num_gold = 0
    num_pred = 0
    for seg, gold in zip(segs, golds):
        seg_spans = set()
        gold_spans = set()
        seg_offset = 0
        for edu in seg.edus():
            seg_spans.add((seg_offset, seg_offset+len(edu.text)-1))
            seg_offset += len(edu.text)
        gold_offset = 0
        for edu in gold.edus():
            gold_spans.add((gold_offset, gold_offset+len(edu.text)-1))
            gold_offset += len(edu.text)
        num_corr += len(seg_spans & gold_spans)
        num_gold += len(gold_spans)
        num_pred += len(seg_spans)
    precision = num_corr / num_pred if num_pred > 0 else 0
    recall = num_corr / num_gold if num_gold > 0 else 0
    if precision + recall == 0:
        f1 = 0.
    else:
        f1 = 2. * precision * recall / (precision + recall)
    return num_gold, num_pred, num_corr, precision, recall, f1


def gen_edu_report(score):
    # num_corr, num_gold, num_pred, precision, recall, f1
    report = '\n'
    report += 'gold     pred     corr     precision    recall     f1\n'
    report += '----------------------------------------------------------\n'
    report += '%-4d     %-4d     %-4d     %-.3f        %-.3f      %-.3f\n' % score
    return report


def rst_parse_eval(parses, golds):
    return parse_eval(parses, golds, average="micro", strict=False, binarize=True)


def parse_eval(parses, golds, average="macro", strict=True, binarize=True):
    """
    :param parses: list of inferenced trees
    :param golds: list of gold standard trees
    :param average: "micro"|"macro"
                    micro average means scores are compute globally, each relation is an instance
                    macro average means score of each tree is compute firstly and then mean of these scores are returned
    :param strict: if set True, the inner node of the tree count as a correct node only if all of it's child have
                   correct boundaries, otherwise, the start offset and end offset of it's boundary are consernded
    :param binarize: binarize tree in parse and gold tree before evaluation
    :return: scores of span, nuclear, relation, full(span+nuclear+relation all correct)
             each score is a triple (recall, precision, f1)
    """
    if len(parses) != len(golds):
        raise ValueError("number of parsed trees should equal to gold standards!")

    num_parse = np.zeros(len(parses), dtype=np.float)
    num_gold = np.zeros(len(golds), dtype=np.float)
    num_corr_span = np.zeros(len(parses), dtype=np.float)
    num_corr_nuc = np.zeros(len(parses), dtype=np.float)
    num_corr_ctype = np.zeros(len(parses), dtype=np.float)
    num_corr_ftype = np.zeros(len(parses), dtype=np.float)
    num_corr_cfull = np.zeros(len(parses), dtype=np.float)
    num_corr_ffull = np.zeros(len(parses), dtype=np.float)

    for i, (parse, gold) in enumerate(zip(parses, golds)):
        parse_quads = factorize_tree(parse, strict, binarize)
        gold_quads = factorize_tree(gold, strict, binarize)
        parse_dict = {quad[0]: quad for quad in parse_quads}
        gold_dict = {quad[0]: quad for quad in gold_quads}

        num_parse[i] = len(parse_quads)
        num_gold[i] = len(gold_quads)

        parse_spans = set(parse_dict.keys())
        gold_spans = set(gold_dict.keys())
        corr_spans = gold_spans & parse_spans
        num_corr_span[i] = len(corr_spans)

        for span in corr_spans:
            # nuclear
            if parse_dict[span][1] == gold_dict[span][1]:
                num_corr_nuc[i] += 1
            # coarse relation
            if parse_dict[span][2] == gold_dict[span][2]:
                num_corr_ctype[i] += 1
            # fine relation
            if parse_dict[span][3] == gold_dict[span][3]:
                num_corr_ftype[i] += 1
            # both nuclear and coarse relation
            if parse_dict[span][1] == gold_dict[span][1] and parse_dict[span][2] == gold_dict[span][2]:
                num_corr_cfull[i] += 1
            # both nuclear and fine relation
            if parse_dict[span][1] == gold_dict[span][1] and parse_dict[span][3] == gold_dict[span][3]:
                num_corr_ffull[i] += 1

    span_score = f1_score(num_corr_span, num_gold, num_parse, average=average)
    nuc_score = f1_score(num_corr_nuc, num_gold, num_parse, average=average)
    ctype_score = f1_score(num_corr_ctype, num_gold, num_parse, average=average)
    ftype_score = f1_score(num_corr_ftype, num_gold, num_parse, average=average)
    cfull_score = f1_score(num_corr_cfull, num_gold, num_parse, average=average)
    ffull_score = f1_score(num_corr_ffull, num_gold, num_parse, average=average)

    # return span_score, nuc_score, ctype_score, ftype_score, cfull_score, ffull_score
    return span_score, nuc_score, ftype_score, ffull_score


def parse_eval_all_info(parses, golds, average="macro", strict=True, binarize=True):
    """
    :param parses: list of inferenced trees
    :param golds: list of gold standard trees
    :param average: "micro"|"macro"
                    micro average means scores are compute globally, each relation is an instance
                    macro average means score of each tree is compute firstly and then mean of these scores are returned
    :param strict: if set True, the inner node of the tree count as a correct node only if all of it's child have
                   correct boundaries, otherwise, the start offset and end offset of it's boundary are consernded
    :param binarize: binarize tree in parse and gold tree before evaluation
    :return: scores of span, nuclear, relation, full(span+nuclear+relation all correct)
             each score is a triple (recall, precision, f1)
    """
    rel_pre_true, rel_gold, rel_pre_all = dict(), dict(), dict()
    nucl_pre_true, nucl_gold = dict(), dict()
    if len(parses) != len(golds):
        raise ValueError("number of parsed trees should equal to gold standards!")

    num_parse = np.zeros(len(parses), dtype=np.float)
    num_gold = np.zeros(len(golds), dtype=np.float)
    num_corr_span = np.zeros(len(parses), dtype=np.float)
    num_corr_nuc = np.zeros(len(parses), dtype=np.float)
    num_corr_ctype = np.zeros(len(parses), dtype=np.float)
    num_corr_ftype = np.zeros(len(parses), dtype=np.float)
    num_corr_cfull = np.zeros(len(parses), dtype=np.float)
    num_corr_ffull = np.zeros(len(parses), dtype=np.float)

    for i, (parse, gold) in enumerate(zip(parses, golds)):
        parse_quads = factorize_tree(parse, strict, binarize)
        gold_quads = factorize_tree(gold, strict, binarize)
        parse_dict = {quad[0]: quad for quad in parse_quads}
        gold_dict = {quad[0]: quad for quad in gold_quads}

        num_parse[i] = len(parse_quads)
        num_gold[i] = len(gold_quads)

        parse_spans = set(parse_dict.keys())
        gold_spans = set(gold_dict.keys())
        corr_spans = gold_spans & parse_spans
        num_corr_span[i] = len(corr_spans)

        for span in corr_spans:
            # nuclear
            if parse_dict[span][1] == gold_dict[span][1]:
                num_corr_nuc[i] += 1
                nucl_pre_true[parse_dict[span][1]] = 1 if parse_dict[span][1] not in nucl_pre_true.keys() \
                    else nucl_pre_true[parse_dict[span][1]] + 1
            nucl_gold[gold_dict[span][1]] = 1 if gold_dict[span][1] not in nucl_gold.keys() \
                else nucl_gold[gold_dict[span][1]] + 1
            # coarse relation
            if parse_dict[span][2] == gold_dict[span][2]:
                num_corr_ctype[i] += 1
            # fine relation
            if parse_dict[span][3] == gold_dict[span][3]:
                num_corr_ftype[i] += 1
                rel_pre_true[parse_dict[span][3]] = 1 if parse_dict[span][3] not in rel_pre_true.keys() \
                    else rel_pre_true[parse_dict[span][3]] + 1
            rel_gold[gold_dict[span][3]] = 1 if gold_dict[span][3] not in rel_gold.keys() \
                else rel_gold[gold_dict[span][3]] + 1
            rel_pre_all[parse_dict[span][3]] = 1 if parse_dict[span][3] not in rel_pre_all.keys() \
                else rel_pre_all[parse_dict[span][3]] + 1
            # both nuclear and coarse relation
            if parse_dict[span][1] == gold_dict[span][1] and parse_dict[span][2] == gold_dict[span][2]:
                num_corr_cfull[i] += 1
            # both nuclear and fine relation
            if parse_dict[span][1] == gold_dict[span][1] and parse_dict[span][3] == gold_dict[span][3]:
                num_corr_ffull[i] += 1

    span_score = f1_score(num_corr_span, num_gold, num_parse, average=average)
    nuc_score = f1_score(num_corr_nuc, num_gold, num_parse, average=average)
    ctype_score = f1_score(num_corr_ctype, num_gold, num_parse, average=average)
    ftype_score = f1_score(num_corr_ftype, num_gold, num_parse, average=average)
    cfull_score = f1_score(num_corr_cfull, num_gold, num_parse, average=average)
    ffull_score = f1_score(num_corr_ffull, num_gold, num_parse, average=average)

    # return span_score, nuc_score, ctype_score, ftype_score, cfull_score, ffull_score
    info_all = ((rel_pre_true, rel_gold, rel_pre_all), (nucl_pre_true, nucl_gold))
    return (span_score, nuc_score, ftype_score, ffull_score), info_all


def gen_parse_report(span_score, nuc_score, ctype_score, ftype_score, cfull_score, ffull_score):
    report = '\n'
    report += '                 precision    recall    f1\n'
    report += '---------------------------------------------\n'
    report += 'span             %5.3f        %5.3f     %5.3f\n' % span_score
    report += 'nuclear          %5.3f        %5.3f     %5.3f\n' % nuc_score
    report += 'ctype            %5.3f        %5.3f     %5.3f\n' % ctype_score
    report += 'cfull            %5.3f        %5.3f     %5.3f\n' % cfull_score
    report += 'ftype            %5.3f        %5.3f     %5.3f\n' % ftype_score
    report += 'ffull            %5.3f        %5.3f     %5.3f\n' % ffull_score
    report += '\n'
    return report


def nuclear_eval(parses, golds, strict=True, binarize=True):
    if len(parses) != len(golds):
        raise ValueError("number of parsed trees should equal to gold standards!")

    num_nuc_parse = defaultdict(int)
    num_nuc_gold = defaultdict(int)
    num_nuc_corr = defaultdict(int)

    for i, (parse, gold) in enumerate(zip(parses, golds)):
        parse_quads = factorize_tree(parse, strict, binarize)
        gold_quads = factorize_tree(gold, strict, binarize)
        parse_dict = {quad[0]: quad for quad in parse_quads}
        gold_dict = {quad[0]: quad for quad in gold_quads}

        parse_spans = set(parse_dict.keys())
        gold_spans = set(gold_dict.keys())
        corr_spans = gold_spans & parse_spans

        for quad in parse_quads:
            num_nuc_parse[quad[1]] += 1
        for quad in gold_quads:
            num_nuc_gold[quad[1]] += 1
        for span in corr_spans:
            if parse_dict[span][1] == gold_dict[span][1]:
                num_nuc_corr[parse_dict[span][1]] += 1

    scores = []
    for nuc_type in set(chain(num_nuc_parse.keys(), num_nuc_gold.keys(), num_nuc_corr.keys())):
        corr = num_nuc_corr[nuc_type]
        pred = num_nuc_parse[nuc_type]
        gold = num_nuc_gold[nuc_type]
        precision = corr / pred if pred > 0 else 0
        recall = corr / gold if gold > 0 else 0
        if precision + recall == 0:
            f1 = 0.
        else:
            f1 = 2. * precision * recall / (precision + recall)
        scores.append((nuc_type, gold, pred, corr, precision, recall, f1))
    return scores


def relation_eval(parses, golds, strict=True, binarize=True):
    if len(parses) != len(golds):
        raise ValueError("number of parsed trees should equal to gold standards!")

    num_crel_parse = defaultdict(int)
    num_crel_gold = defaultdict(int)
    num_crel_corr = defaultdict(int)
    num_frel_parse = defaultdict(int)
    num_frel_gold = defaultdict(int)
    num_frel_corr = defaultdict(int)

    for i, (parse, gold) in enumerate(zip(parses, golds)):
        parse_quads = factorize_tree(parse, strict, binarize)
        gold_quads = factorize_tree(gold, strict, binarize)
        parse_dict = {quad[0]: quad for quad in parse_quads}
        gold_dict = {quad[0]: quad for quad in gold_quads}

        parse_spans = set(parse_dict.keys())
        gold_spans = set(gold_dict.keys())
        corr_spans = gold_spans & parse_spans

        for quad in parse_quads:
            num_crel_parse[quad[2]] += 1
            num_frel_parse[quad[3]] += 1
        for quad in gold_quads:
            num_crel_gold[quad[2]] += 1
            num_frel_gold[quad[3]] += 1
        for span in corr_spans:
            if parse_dict[span][2] == gold_dict[span][2]:
                num_crel_corr[parse_dict[span][2]] += 1
            if parse_dict[span][3] == gold_dict[span][3]:
                num_frel_corr[parse_dict[span][3]] += 1

    crel_scores = []
    frel_scores = []
    for nuc_type in set(chain(num_crel_parse.keys(), num_crel_gold.keys(), num_crel_corr.keys())):
        corr = num_crel_corr[nuc_type]
        pred = num_crel_parse[nuc_type]
        gold = num_crel_gold[nuc_type]
        precision = corr / pred if pred > 0 else 0
        recall = corr / gold if gold > 0 else 0
        if precision + recall == 0:
            f1 = 0.
        else:
            f1 = 2. * precision * recall / (precision + recall)
        crel_scores.append((nuc_type, gold, pred, corr, precision, recall, f1))
    for nuc_type in set(chain(num_frel_parse.keys(), num_frel_gold.keys(), num_frel_corr.keys())):
        corr = num_frel_corr[nuc_type]
        pred = num_frel_parse[nuc_type]
        gold = num_frel_gold[nuc_type]
        precision = corr / pred if pred > 0 else 0
        recall = corr / gold if gold > 0 else 0
        if precision + recall == 0:
            f1 = 0.
        else:
            f1 = 2. * precision * recall / (precision + recall)
        frel_scores.append((nuc_type, gold, pred, corr, precision, recall, f1))
    return crel_scores, frel_scores


def gen_category_report(scores):
    report = '\n'
    report += 'type    gold       pred       corr   precision    recall        f1\n'
    report += '--------------------------------------------------------------------\n'
    for score in scores:
        report += '%-4s   %5d      %5d      %5d       %5.3f     %5.3f     %5.3f\n' % score
    return report


def factorize_tree(tree, strict=False, binarize=True):
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
                    if strict:
                        span = left, right
                    else:
                        span = left[0], right[1]
                    quads.add((span, root.nuclear, root.ctype, root.ftype))
                    children_spans.append((left[0], right[1]))
            else:
                if strict:
                    span = children_spans
                else:
                    span = children_spans[0][0], children_spans[-1][1]
                quads.add((span, root.nuclear, root.ctype, root.ftype))
            return [(children_spans[0][0], children_spans[-1][1])]

    factorize(tree.root_relation())
    return quads


def height_eval(parses, golds, strict=True, binarize=True):
    gold_heights = []
    corr_heights = []

    for i, (parse, gold) in enumerate(zip(parses, golds)):
        parse_quads = factorize_tree(parse, strict, binarize)
        gold_quads = factorize_tree(gold, strict, binarize)
        parse_dict = {quad[0]: quad for quad in parse_quads}
        gold_dict = {quad[0]: quad for quad in gold_quads}

        parse_spans = set(parse_dict.keys())
        gold_spans = set(gold_dict.keys())
        corr_spans = gold_spans & parse_spans

        span_heights = factorize_span_height(gold, strict=strict, binarize=binarize)
        for span, height in span_heights.items():
            gold_heights.append(height)
            if span in corr_spans:
                corr_heights.append(height)

    gold_count = Counter(gold_heights)
    corr_count = Counter(corr_heights)
    height_scores = []
    for height in sorted(gold_count.keys()):
        precision = float(corr_count[height]) / (float(gold_count[height]) + 1e-8)
        height_scores.append((height, gold_count[height], corr_count[height], precision))
    return height_scores


def gen_height_report(scores):
    report = '\n'
    report += 'height    gold       corr    precision\n'
    report += '------------------------------------------\n'
    for score in scores:
        report += '%-4s    %5d      %5d     %5.3f\n' % score
    return report


def factorize_span_height(tree, strict=False, binarize=True):
    span_height = {}

    def factorize(root, offset=0):
        if isinstance(root, EDU):
            return 0, [(offset, offset + len(root.text))]  # height, child_spans
        elif isinstance(root, Sentence):
            children_spans = []
            max_height = 0
            for child in root:
                height, spans = factorize(child, offset)
                children_spans.extend(spans)
                offset = spans[-1][1]
                max_height = height if height > max_height else max_height
            return max_height, children_spans
        elif isinstance(root, Relation):
            children_spans = []
            max_height = 0
            for child in root:
                height, spans = factorize(child, offset)
                children_spans.extend(spans)
                offset = spans[-1][1]
                max_height = height if height > max_height else max_height
            if binarize:
                while len(children_spans) >= 2:
                    right = children_spans.pop()
                    left = children_spans.pop()
                    if strict:
                        span = left, right
                    else:
                        span = left[0], right[1]
                    max_height += 1
                    span_height[span] = max_height
                    children_spans.append((left[0], right[1]))
            else:
                if strict:
                    span = children_spans
                else:
                    span = children_spans[0][0], children_spans[-1][1]
                max_height += 1
                span_height[span] = max_height
            return max_height, [(children_spans[0][0], children_spans[-1][1])]

    factorize(tree.root_relation())
    return span_height


def f1_score(num_corr, num_gold, num_pred, average="micro"):
    if average == "micro":
        precision = np.nan_to_num(num_corr.sum() / num_pred.sum())
        recall = np.nan_to_num(num_corr.sum() / num_gold.sum())
    elif average == "macro":
        precision = (np.nan_to_num(num_corr / (num_pred + 1e-10))).mean()
        recall = (np.nan_to_num(num_corr / num_gold)).mean()
    else:
        raise ValueError("unsupported average mode '%s'" % average)
    if precision + recall == 0:
        f1 = 0.
    else:
        f1 = 2. * precision * recall / (precision + recall)
    return precision, recall, f1
