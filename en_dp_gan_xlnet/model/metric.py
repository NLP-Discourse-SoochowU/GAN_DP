# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date: 2019.7.16
@Description: micro + standard parseval
"""
import numpy as np
from config import *
from path_config import MODELS2SAVE


class Metrics(object):
    def __init__(self, log_file=None):
        self.log_file = log_file

        # micro data save. s n r f
        self.true_all = [0., 0., 0., 0.]
        self.span_all = 0.

        self.dev_f_max, self.test_f_max = 0., 0.
        self.dev_s_max, self.test_s_max = 0., 0.
        self.dev_n_max, self.test_n_max = 0., 0.
        self.dev_r_max, self.test_r_max = 0., 0.

        self.dev_f_m_scores = [0., 0., 0., 0.]
        self.test_f_m_scores, self.test_s_m_scores = [0., 0., 0., 0.], [0., 0., 0., 0.]
        self.test_n_m_scores, self.test_r_m_scores = [0., 0., 0., 0.], [0., 0., 0., 0.]

        self.ns_pre, self.ns_gold = [0., 0., 0.], [0., 0., 0.]
        self.rel_pre_true, self.rel_gold, self.rel_pre_all = [0. for _ in range(18)], [0. for _ in range(18)], \
                                                             [0. for _ in range(18)]
        self.ns_scores = [0., 0., 0.]
        self.rel_scores_p, self.rel_scores_r, self.rel_scores_f = [0. for _ in range(18)], [0. for _ in range(18)], \
                                                                  [0. for _ in range(18)]

    def init_all(self):
        self.true_all, self.span_all = [0., 0., 0., 0.], 0.
        self.ns_pre, self.ns_gold = [0., 0., 0.], [0., 0., 0.]
        self.rel_pre_true, self.rel_gold, self.rel_pre_all = [0. for _ in range(18)], [0. for _ in range(18)], \
                                                             [0. for _ in range(18)]

    def eval_(self, goldtrees, predtrees, model=None, type_="dev"):
        self.init_all()
        for idx in range(len(goldtrees)):
            gold_ids = self.get_all_span_info(goldtrees[idx])
            pred_ids = self.get_all_span_info(predtrees[idx])
            self.eval_all(gold_ids, pred_ids)
        better = self.report(model=model, type_=type_, predtrees=predtrees)
        return better

    def eval_all(self, gold_ids, pred_ids):
        gold_s_ids, gold_ns_ids, gold_rel_ids = gold_ids
        pred_s_ids, pred_ns_ids, pred_rel_ids = pred_ids

        # span
        allspan = [span for span in gold_s_ids if span in pred_s_ids]
        allspan_gold_idx = [gold_s_ids.index(span) for span in allspan]
        allspan_pred_idx = [pred_s_ids.index(span) for span in allspan]

        # ns
        all_gold_ns = [gold_ns_ids[idx] for idx in allspan_gold_idx]
        all_pred_ns = [pred_ns_ids[idx] for idx in allspan_pred_idx]

        # rel
        all_gold_rel = [gold_rel_ids[idx] for idx in allspan_gold_idx]
        all_pred_rel = [pred_rel_ids[idx] for idx in allspan_pred_idx]

        # macro & micro
        span_len = float(len(gold_s_ids))
        self.compute_parseval_back((allspan, span_len), (all_gold_ns, all_pred_ns),
                                   (all_gold_rel, all_pred_rel), gold_ns_ids, gold_rel_ids, pred_rel_ids)

    def compute_parseval_back(self, span_, ns_, rel_, all_ns, all_rel, all_rel_pre):
        """ standard parseval: macro & micro
            Get a full computation version.
        """
        ns_equal = np.equal(ns_[0], ns_[1])
        rel_equal = np.equal(rel_[0], rel_[1])

        f_equal = [ns_equal[idx] and rel_equal[idx] for idx in range(len(ns_equal))]
        s_pred, ns_pred, rel_pred, f_pred = len(span_[0]), sum(ns_equal), sum(rel_equal), sum(f_equal)
        # micro
        self.true_all[0] += s_pred
        self.true_all[1] += ns_pred
        self.true_all[2] += rel_pred
        self.true_all[3] += f_pred
        self.span_all += span_[1]

        for rel_one_ in all_rel_pre:
            self.rel_pre_all[rel_one_] += 1.

        for ns_one, rel_one in zip(all_ns, all_rel):
            self.ns_gold[ns_one] += 1.
            self.rel_gold[rel_one] += 1.

        for idx in range(len(ns_[0])):
            tmp_ns_gold, tmp_ns_pre = ns_[0][idx], ns_[1][idx]
            if tmp_ns_pre == tmp_ns_gold:
                self.ns_pre[tmp_ns_pre] += 1.

            tmp_rel_gold, tmp_rel_pre = rel_[0][idx], rel_[1][idx]
            if tmp_rel_pre == tmp_rel_gold:
                self.rel_pre_true[tmp_rel_pre] += 1.

    @staticmethod
    def get_all_span_info(tree_):
        span_ids, ns_ids, rel_ids = [], [], []
        for node in tree_.nodes:
            if node.left_child is not None and node.right_child is not None:
                span_ids.append(node.temp_edu_span)
                ns_ids.append(nucl2ids[node.child_NS_rel])
                rel_ids.append(coarse2ids[node.child_rel])
        return span_ids, ns_ids, rel_ids

    def report(self, model, type_="dev", predtrees=None):
        p_span, p_ns, p_rel, p_f = (self.true_all[idx] / self.span_all for idx in range(4))  # micro

        f_max = self.test_f_max if type_ == "test" else self.dev_f_max
        s_max = self.test_s_max if type_ == "test" else self.dev_s_max
        n_max = self.test_n_max if type_ == "test" else self.dev_n_max
        r_max = self.test_r_max if type_ == "test" else self.dev_r_max
        better = self.update_all_max(p_span, p_ns, p_rel, p_f, f_max, s_max, n_max, r_max, type_)

        self.save_best_models(p_f, f_max, p_span, s_max, p_ns, n_max, p_rel, r_max, model, type_, predtrees)
        return better

    def update_all_max(self, span_pre, nucl_pre, rel_pre, f_pre, f_max, s_max, n_max, r_max, type_):
        better = False
        modes = [(span_pre > s_max), (nucl_pre > n_max), (rel_pre > r_max), (f_pre > f_max)]
        if modes[0] or modes[1] or modes[2] or modes[3]:
            if type_ == "test":
                if modes[0]:
                    self.test_s_max = span_pre
                    self.test_s_m_scores = [span_pre, nucl_pre, rel_pre, f_pre]
                if modes[1]:
                    self.test_n_max = nucl_pre
                    self.test_n_m_scores = [span_pre, nucl_pre, rel_pre, f_pre]
                if modes[2]:
                    self.test_r_max = rel_pre
                    self.test_r_m_scores = [span_pre, nucl_pre, rel_pre, f_pre]
                if modes[3]:
                    self.test_f_max = f_pre
                    self.test_f_m_scores = [span_pre, nucl_pre, rel_pre, f_pre]

                    self.ns_scores = [self.ns_pre[idx] / self.ns_gold[idx] for idx in range(3)]
                    for rel_idx in range(18):
                        c_ = self.rel_pre_true[rel_idx]
                        g_ = self.rel_gold[rel_idx]
                        h_ = self.rel_pre_all[rel_idx]
                        p_b = 0. if h_ == 0 else c_ / h_
                        r_b = 0. if g_ == 0 else c_ / g_
                        f_b = 0. if (g_ + h_) == 0 else (2 * c_) / (g_ + h_)
                        self.rel_scores_p[rel_idx] = p_b
                        self.rel_scores_r[rel_idx] = r_b
                        self.rel_scores_f[rel_idx] = f_b
            else:
                better = True
                self.dev_f_max = f_pre
                self.dev_f_m_scores = [span_pre, nucl_pre, rel_pre, f_pre]
        return better

    def save_best_models(self, f_pre, f_max, p_span, s_max, p_ns, n_max, p_rel, r_max, model, type_, predtrees):
        if SAVE_MODEL:
            f_file_name = "/test_f_max_model.pth" if type_ == "test" else "/dev_f_max_model.pth"
            f_best_trees_parsed = "/test_f_trees.pkl" if type_ == "test" else "/dev_f_trees.pkl"
            if f_pre > f_max:
                self.save_model(file_name=f_file_name, model=model)
                self.save_trees(file_name=f_best_trees_parsed, trees=predtrees)
            s_file_name = "/test_s_max_model.pth" if type_ == "test" else "/dev_s_max_model.pth"
            s_best_trees_parsed = "/test_s_trees.pkl" if type_ == "test" else "/dev_s_trees.pkl"
            if p_span > s_max:
                self.save_model(file_name=s_file_name, model=model)
                self.save_trees(file_name=s_best_trees_parsed, trees=predtrees)
            n_file_name = "/test_n_max_model.pth" if type_ == "test" else "/dev_n_max_model.pth"
            n_best_trees_parsed = "/test_n_trees.pkl" if type_ == "test" else "/dev_n_trees.pkl"
            if p_ns > n_max:
                self.save_model(file_name=n_file_name, model=model)
                self.save_trees(file_name=n_best_trees_parsed, trees=predtrees)
            r_file_name = "/test_r_max_model.pth" if type_ == "test" else "/dev_r_max_model.pth"
            r_best_trees_parsed = "/test_r_trees.pkl" if type_ == "test" else "/dev_r_trees.pkl"
            if p_rel > r_max:
                self.save_model(file_name=r_file_name, model=model)
                self.save_trees(file_name=r_best_trees_parsed, trees=predtrees)

    @staticmethod
    def save_model(file_name, model):
        dir2save = MODELS2SAVE + "/v" + str(VERSION) + "_set" + str(SET)
        safe_mkdir(MODELS2SAVE)
        safe_mkdir(dir2save)
        save_path = dir2save + file_name
        torch.save(model, save_path)

    @staticmethod
    def save_trees(file_name, trees):
        dir2save = MODELS2SAVE + "/v" + str(VERSION) + "_set" + str(SET)
        safe_mkdir(MODELS2SAVE)
        safe_mkdir(dir2save)
        save_path = dir2save + file_name
        save_data(trees, save_path)

    def output_report(self, report_info=None):
        for info in report_info:
            print_(info, self.log_file)

    def get_scores(self):
        """ (mi, inner_mi, ma, inner_ma)
        """
        test_out = [self.test_s_m_scores, self.test_n_m_scores, self.test_r_m_scores, self.test_f_m_scores]
        report_info = ["==============================================",
                       "(VAL_) [" + str(self.dev_f_m_scores[0]) + ", " + str(self.dev_f_m_scores[1])
                       + ", " + str(self.dev_f_m_scores[2]) + ", " + str(self.dev_f_m_scores[3]) + "]",
                       "(TEST Bare) [" + str(test_out[0][0]) + ", " + str(test_out[0][1]) + ", " +
                       str(test_out[0][2]) + ", " + str(test_out[0][3]) + "]",
                       "(TEST N) [" + str(test_out[1][0]) + ", " + str(test_out[1][1]) + ", " +
                       str(test_out[1][2]) + ", " + str(test_out[1][3]) + "]",
                       "(TEST R) [" + str(test_out[2][0]) + ", " + str(test_out[2][1]) + ", " +
                       str(test_out[2][2]) + ", " + str(test_out[2][3]) + "]",
                       "(TEST F) [" + str(test_out[3][0]) + ", " + str(test_out[3][1]) + ", " +
                       str(test_out[3][2]) + ", " + str(test_out[3][3]) + "]"
                       , "(NN, NS, SN): ", " ".join([str(item) for item in self.ns_scores]),
                       "(18 RELs P): ", " ".join([str(item) for item in self.rel_scores_p]),
                       "(18 RELs R): ", " ".join([str(item) for item in self.rel_scores_r]),
                       "(18 RELs F): ", " ".join([str(item) for item in self.rel_scores_f])]
        return report_info
