# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
import re

leaf_parttern = r' *\( \w+ \(leaf .+'
leaf_re = re.compile(leaf_parttern)
node_parttern = r' *\( \w+ \(span .+'
node_re = re.compile(node_parttern)
end_parttern = r'\s*\)\s*'
end_re = re.compile(end_parttern)
nodetype_parttern = r' *\( (\w+) .+'
type_re = re.compile(nodetype_parttern)
rel_parttern = r' *\( \w+ \(.+\) \(rel2par ([\w-]+).+'
rel_re = re.compile(rel_parttern)
node_leaf_parttern = r' *\( \w+ \((\w+) \d+.*\).+'
node_leaf_re = re.compile(node_leaf_parttern)
upper_parttern = r'[a-z].*'
upper_re = re.compile(upper_parttern)
