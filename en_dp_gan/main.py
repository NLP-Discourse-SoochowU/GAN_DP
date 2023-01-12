# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
import logging
from sys import argv
from model.stacked_parser_tdt_gan3.trainer import Trainer


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_desc = argv[1] if len(argv) >= 2 else "no message."
    trainer = Trainer()
    trainer.train(test_desc)
