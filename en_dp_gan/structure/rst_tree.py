from util.patterns import *
import copy
from util.file_util import load_data
from path_config import REL_raw2coarse


def get_blank(line):
    count = 0
    while line[count] == " ":
        count += 1
    return count


class rst_tree:
    def __init__(self, type_=None, l_ch=None, r_ch=None, p_node=None, child_rel=None, rel=None, ch_ns_rel="",
                 lines_list=None, temp_line=" ", file_name=None, temp_edu=None, temp_edu_span=None,
                 temp_edu_ids=None, temp_pos_ids=None, tmp_conn_ids=None, temp_edu_heads=None,
                 temp_edu_has_center_word=None, raw_rel=False, temp_edu_freq=None, temp_edus_count=0,
                 tmp_edu_emlo_emb=None, right_branch=False, attn_assigned=None, temp_edu_boundary=None, node_height=0):
        self.file_name = file_name
        # Satellite Nucleus Root
        self.type = type_
        self.left_child = l_ch
        self.right_child = r_ch
        self.parent_node = p_node
        self.rel = rel
        self.child_rel = child_rel
        self.child_NS_rel = ch_ns_rel
        self.lines_list = lines_list
        self.temp_line = temp_line
        self.temp_edu = temp_edu
        self.temp_edu_span = temp_edu_span
        self.temp_edu_ids = temp_edu_ids
        self.temp_edu_emlo_emb = tmp_edu_emlo_emb
        self.temp_edus_count = temp_edus_count
        self.temp_edu_freq = temp_edu_freq
        self.temp_pos_ids = temp_pos_ids
        self.temp_edu_conn_ids = tmp_conn_ids
        self.temp_edu_heads = temp_edu_heads
        self.temp_edu_has_center_word = temp_edu_has_center_word
        self.edu_node = []
        self.rel_raw2coarse = load_data(REL_raw2coarse)
        self.raw_rel = raw_rel
        self.feature_label = None
        self.inner_sent = False
        self.edu_node_boundary = temp_edu_boundary
        self.right_branch = right_branch
        self.attn_assigned = attn_assigned
        self.node_height = node_height

    def __copy__(self):
        new_obj = rst_tree(type_=self.type, l_ch=copy.copy(self.left_child), r_ch=copy.copy(self.right_child),
                           p_node=copy.copy(self.parent_node), child_rel=self.child_rel, rel=self.rel,
                           ch_ns_rel=self.child_NS_rel, lines_list=self.lines_list[:], temp_line=self.temp_line,
                           file_name=self.file_name, tmp_conn_ids=None, raw_rel=self.raw_rel)
        return new_obj

    def append(self, root):
        pass

    def get_type(self, line):
        pass

    def create_tree(self, temp_line_num, p_node_=None):
        if temp_line_num > len(self.lines_list):
            return
        line = self.lines_list[temp_line_num]
        count_blank = get_blank(line)
        child_list = []
        while temp_line_num < len(self.lines_list):
            line = self.lines_list[temp_line_num]
            if get_blank(line) == count_blank:
                temp_line_num += 1
                node_type = type_re.findall(line)[0]
                if self.raw_rel:
                    node_new = rst_tree(type_=node_type, temp_line=line, rel=rel_re.findall(line)[0],
                                        lines_list=self.lines_list, raw_rel=self.raw_rel)
                else:
                    node_new = rst_tree(type_=node_type, temp_line=line, raw_rel=self.raw_rel,
                                        rel=self.rel_raw2coarse[rel_re.findall(line)[0]], lines_list=self.lines_list)
                if node_re.match(line):
                    temp_line_num = node_new.create_tree(temp_line_num=temp_line_num, p_node_=node_new)
                elif leaf_re.match(line):
                    pass
                child_list.append(node_new)
            else:
                while temp_line_num < len(self.lines_list) and end_re.match(self.lines_list[temp_line_num]):
                    temp_line_num += 1
                break
        while len(child_list) > 2:
            temp_r = child_list.pop()
            temp_l = child_list.pop()
            if not temp_l.rel == "span":
                new_node = rst_tree(type_="Nucleus", r_ch=temp_r, l_ch=temp_l, ch_ns_rel="NN", child_rel=temp_l.rel,
                                    rel=temp_l.rel, temp_line="<new created line>", lines_list=self.lines_list,
                                    raw_rel=self.raw_rel)
            if not temp_r.rel == "span":
                new_node = rst_tree(type_="Nucleus", r_ch=temp_r, l_ch=temp_l, ch_ns_rel="NN", child_rel=temp_r.rel,
                                    rel=temp_r.rel, temp_line="<new created line>", lines_list=self.lines_list,
                                    raw_rel=self.raw_rel)
            new_node.temp_line = "<new created line>"
            temp_r.parent_node = new_node
            temp_l.parent_node = new_node
            child_list.append(new_node)

        self.right_child = child_list.pop()
        self.left_child = child_list.pop()
        self.right_child.parent_node = p_node_
        self.left_child.parent_node = p_node_
        if not self.right_child.rel == "span":
            self.child_rel = self.right_child.rel
        if not self.left_child.rel == "span":
            self.child_rel = self.left_child.rel
        self.child_NS_rel += self.left_child.type[0] + self.right_child.type[0]
        return temp_line_num

    def config_edus(self, temp_node, e_list, e_span_list, e_ids_list, e_tags_list, e_node=None, e_headwords_list=None,
                    e_cent_word_list=None, total_e_num=None, e_bound_list=None, e_emlo_list=None):
        if e_node is None:
            e_node = []
        if self.right_child is not None and self.left_child is not None:
            self.left_child.config_edus(self.left_child, e_list, e_span_list, e_ids_list, e_tags_list, e_node,
                                        e_headwords_list, e_cent_word_list, total_e_num, e_bound_list, e_emlo_list)
            self.right_child.config_edus(self.right_child, e_list, e_span_list, e_ids_list, e_tags_list, e_node,
                                         e_headwords_list, e_cent_word_list, total_e_num, e_bound_list, e_emlo_list)
            self.temp_edu = self.left_child.temp_edu + " " + self.right_child.temp_edu
            self.temp_edu_span = (self.left_child.temp_edu_span[0], self.right_child.temp_edu_span[1])
            self.temp_edu_ids = self.left_child.temp_edu_ids + self.right_child.temp_edu_ids
            self.temp_pos_ids = self.left_child.temp_pos_ids + self.right_child.temp_pos_ids
            if self.left_child.inner_sent is False or self.right_child.inner_sent is False:
                self.inner_sent = False
                self.edu_node_boundary = False
            elif self.left_child.edu_node_boundary is False:
                self.inner_sent = True
                if self.right_child.edu_node_boundary is True:
                    self.edu_node_boundary = True
                else:
                    self.edu_node_boundary = False
            else:
                self.inner_sent = False
                self.edu_node_boundary = False
            self.temp_edus_count = self.left_child.temp_edus_count + self.right_child.temp_edus_count
            self.node_height = max(self.left_child.node_height, self.right_child.node_height) + 1

        elif self.right_child is None and self.left_child is None:
            self.temp_edu = e_list.pop(0)
            self.temp_edu_span = e_span_list.pop(0)
            self.temp_edu_ids = e_ids_list.pop(0)
            self.temp_edu_emlo_emb = e_emlo_list.pop(0)
            self.temp_pos_ids = e_tags_list.pop(0)
            self.edu_node_boundary = e_bound_list.pop(0)
            self.inner_sent = True
            self.temp_edus_count = 1
            # dependency
            self.temp_edu_heads = e_headwords_list.pop(0)
            self.temp_edu_has_center_word = e_cent_word_list.pop(0)
            e_node.append(temp_node)

    def config_nodes(self, root):
        if root is None or root.left_child is None or root.right_child is None:
            return
        self.config_nodes(root.left_child)
        self.config_nodes(root.right_child)
        root.temp_edu = root.left_child.temp_edu + " " + root.right_child.temp_edu
        root.temp_edu_span = (root.left_child.temp_edu_span[0], root.right_child.temp_edu_span[1])
        root.temp_edu_ids = root.left_child.temp_edu_ids + root.right_child.temp_edu_ids
        root.temp_pos_ids = root.left_child.temp_pos_ids + root.right_child.temp_pos_ids

    def pre_traverse(self, count=0):
        if self.right_child is not None and self.left_child is not None:
            count_ = self.left_child.pre_traverse(count)
            count_ = self.right_child.pre_traverse(count_)
            return count_
        else:
            return count + 1
