# coding: UTF-8
from abc import abstractmethod

from typing import List
from structure.nodes import Sentence, Paragraph, EDU


class SegmenterI:
    @abstractmethod
    def cut(self, text: str) -> Paragraph:
        raise NotImplemented()

    @abstractmethod
    def cut_sent(self, text: str, sid=None) -> List[Sentence]:
        raise NotImplemented()

    @abstractmethod
    def cut_edu(self, sent: Sentence) -> List[EDU]:
        raise NotImplemented()


class ParserI:
    @abstractmethod
    def parse(self, para: Paragraph) -> Paragraph:
        raise NotImplemented()


class PipelineI:
    @abstractmethod
    def cut_sent(self, text: str) -> Paragraph:
        raise NotImplemented()

    @abstractmethod
    def cut_edu(self, para: Paragraph) -> Paragraph:
        raise NotImplemented()

    @abstractmethod
    def parse(self, para: Paragraph) -> Paragraph:
        raise NotImplemented()

    @abstractmethod
    def full_parse(self, text: str):
        raise NotImplemented()

    def __call__(self, text: str):
        return self.full_parse(text)
