# coding: UTF-8
import re
import os
from nltk.tree import Tree as ParseTree
from util.berkely import BerkeleyParser
from tqdm import tqdm


if __name__ == '__main__':
    ctb_dir = "data/CTB"
    save_dir = "data/CTB_auto"
    encoding = "UTF-8"
    ctb = {}
    s_pat = re.compile(r"<S ID=(?P<sid>\S+?)>(?P<sparse>.*?)</S>", re.M | re.DOTALL)
    parser = BerkeleyParser()
    for file in tqdm(os.listdir(ctb_dir)):
        if os.path.isfile(os.path.join(save_dir, file)):
            continue
        print(file)
        with open(os.path.join(ctb_dir, file), "r", encoding=encoding) as fd:
            doc = fd.read()
        parses = []
        for match in s_pat.finditer(doc):
            sid = match.group("sid")
            sparse = ParseTree.fromstring(match.group("sparse"))
            pairs = [(node[0], node.label()) for node in sparse.subtrees()
                     if node.height() == 2 and node.label() != "-NONE-"]
            words, tags = list(zip(*pairs))
            print(sid, " ".join(words))
            if sid == "5133":
                parse = sparse
            else:
                parse = parser.parse(words, timeout=2000)
            parses.append((sid, parse))
        with open(os.path.join(save_dir, file), "w+", encoding=encoding) as save_fd:
            for sid, parse in parses:
                save_fd.write("<S ID=%s>\n" % sid)
                save_fd.write(str(parse))
                save_fd.write("\n</S>\n")
