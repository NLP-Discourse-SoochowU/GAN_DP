# coding: UTF-8

import os
import pyltp

LTP_DATA_DIR = 'pub/pyltp_models'
path_to_tagger = os.path.join(LTP_DATA_DIR, 'pos.model')
path_to_parser = os.path.join(LTP_DATA_DIR, "parser.model")


class LTPParser:
    def __init__(self):
        self.tagger = pyltp.Postagger()
        self.parser = pyltp.Parser()
        self.tagger.load(path_to_tagger)
        self.parser.load(path_to_parser)

    def __enter__(self):
        self.tagger.load(path_to_tagger)
        self.parser.load(path_to_parser)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tagger.release()
        self.parser.release()

    def parse(self, words):
        tags = self.tagger.postag(words)
        parse = self.parser.parse(words, tags)
        return parse
