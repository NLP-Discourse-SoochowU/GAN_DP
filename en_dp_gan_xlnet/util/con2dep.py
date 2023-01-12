# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""

dep_tuples, dep_idx = [], -1


def con2dep(root):
    global dep_tuples, dep_idx
    dep_tuples, dep_idx = [], -1
    recursive_one(root)
    return dep_tuples


def recursive_one(root):
    global dep_tuples, dep_idx
    if root.left_child is not None:
        idx1 = recursive_one(root.left_child)
        idx2 = recursive_one(root.right_child)
        later = max(idx1, idx2)
        pre = min(idx1, idx2)
        if root.child_NS_rel == "SN":
            dep_tuples.append((later, pre, 0, root.child_rel))
            return idx2
        else:
            dep_tuples.append((later, pre, 1, root.child_rel))
            return idx1
    else:
        dep_idx += 1
        return dep_idx
