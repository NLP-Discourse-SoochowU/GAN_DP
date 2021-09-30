# coding: UTF-8

import os
import re
import pickle
import gzip
import hashlib
import thulac
import tqdm
import logging
from itertools import chain
from nltk.tree import Tree as ParseTree
from structure import Discourse, Sentence, EDU, TEXT, Connective, node_type_filter


# note: this module will print a junk line "Model loaded succeed" to stdio when initializing
thulac = thulac.thulac()
logger = logging.getLogger(__name__)


class CDTB:
    def __init__(self, cdtb_dir, train, validate, test, encoding="UTF-8",
                 ctb_dir=None, ctb_encoding="UTF-8", cache_dir=None, preprocess=False):
        if cache_dir:
            hash_figure = "-".join(map(str, [cdtb_dir, train, validate, test, ctb_dir, preprocess]))
            hash_key = hashlib.md5(hash_figure.encode()).hexdigest()
            cache = os.path.join(cache_dir, hash_key + ".gz")
        else:
            cache = None

        if cache is not None and os.path.isfile(cache):
            with gzip.open(cache, "rb") as cache_fd:
                self.preprocess, self.ctb = pickle.load(cache_fd)
                self.train = pickle.load(cache_fd)
                self.validate = pickle.load(cache_fd)
                self.test = pickle.load(cache_fd)
            logger.info("load cached dataset from %s" % cache)
            return

        self.preprocess = preprocess
        self.ctb = self.load_ctb(ctb_dir, ctb_encoding) if ctb_dir else {}
        self.train = self.load_dir(cdtb_dir, train, encoding=encoding)
        self.validate = self.load_dir(cdtb_dir, validate, encoding=encoding)
        self.test = self.load_dir(cdtb_dir, test, encoding=encoding)

        if preprocess:
            for discourse in tqdm.tqdm(chain(self.train, self.validate, self.test), desc="preprocessing"):
                self.preprocessing(discourse)

        if cache is not None:
            with gzip.open(cache, "wb") as cache_fd:
                pickle.dump((self.preprocess, self.ctb), cache_fd)
                pickle.dump(self.train, cache_fd)
                pickle.dump(self.validate, cache_fd)
                pickle.dump(self.test, cache_fd)
                logger.info("saved cached dataset to %s" % cache)

    def report(self):
        # TODO
        raise NotImplementedError()

    def preprocessing(self, discourse):
        for paragraph in discourse:
            for sentence in paragraph.iterfind(filter=node_type_filter(Sentence)):
                if self.ctb and (sentence.sid is not None) and (sentence.sid in self.ctb):
                    parse = self.ctb[sentence.sid]
                    pairs = [(node[0], node.label()) for node in parse.subtrees()
                             if node.height() == 2 and node.label() != "-NONE-"]
                    words, tags = list(zip(*pairs))
                else:
                    words, tags = list(zip(*thulac.cut(sentence.text)))
                setattr(sentence, "words", list(words))
                setattr(sentence, "tags", list(tags))

                offset = 0
                for textnode in sentence.iterfind(filter=node_type_filter([TEXT, Connective, EDU]),
                                                  terminal=node_type_filter([TEXT, Connective, EDU])):
                    if isinstance(textnode, EDU):
                        edu_words = []
                        edu_tags = []
                        cur = 0
                        for word, tag in zip(sentence.words, sentence.tags):
                            if offset <= cur < cur + len(word) <= offset + len(textnode.text):
                                edu_words.append(word)
                                edu_tags.append(tag)
                            cur += len(word)
                        setattr(textnode, "words", edu_words)
                        setattr(textnode, "tags", edu_tags)
                    offset += len(textnode.text)
        return discourse

    @staticmethod
    def load_dir(path, sub, encoding="UTF-8"):
        train_path = os.path.join(path, sub)
        discourses = []
        for file in os.listdir(train_path):
            file = os.path.join(train_path, file)
            discourse = Discourse.from_xml(file, encoding=encoding)
            discourses.append(discourse)
        return discourses

    @staticmethod
    def load_ctb(ctb_dir, encoding="UTF-8"):
        ctb = {}
        s_pat = re.compile("<S ID=(?P<sid>\S+?)>(?P<sparse>.*?)</S>", re.M | re.DOTALL)
        for file in os.listdir(ctb_dir):
            with open(os.path.join(ctb_dir, file), "r", encoding=encoding) as fd:
                doc = fd.read()
            for match in s_pat.finditer(doc):
                sid = match.group("sid")
                sparse = ParseTree.fromstring(match.group("sparse"))
                ctb[sid] = sparse
        return ctb
