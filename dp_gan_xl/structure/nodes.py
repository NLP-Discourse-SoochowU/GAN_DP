# coding: UTF-8
from xml.etree import ElementTree
from xml.dom import minidom
from nltk.tree import ParentedTree


class BaseNode(ParentedTree):
    def _get_node(self):
        raise DeprecationWarning()

    def _set_node(self, value):
        raise DeprecationWarning()

    def iterfind(self, filter=None, terminal=None, prior=False):
        yield from BaseNode.cls_iterfind(self, filter, terminal, prior)

    @classmethod
    def cls_iterfind(cls, root, filter=None, terminal=None, prior=False):
        if filter is not None and not callable(filter):
            raise ValueError("filter should be callable or None")

        if prior and (filter is None or filter(root)):
            yield root

        if terminal is None or not terminal(root):
            for node in root:
                if isinstance(node, BaseNode):
                    yield from cls.cls_iterfind(node, filter, terminal, prior)

        if not prior and (filter is None or filter(root)):
            yield root

    def __hash__(self):
        return id(self)


class TEXT(BaseNode):
    def __init__(self, text):
        super(TEXT, self).__init__("TEXT", [text])
        self.text = text


class Connective(TEXT):
    def __init__(self, text, cid):
        super(Connective, self).__init__(text)
        self.cid = cid
        self.set_label("CONNECTIVE")

    def label(self):
        return BaseNode.label(self) + "(%s)" % str(self.cid)


class EDU(BaseNode):
    def __init__(self, children=None):
        children = children or []
        super(EDU, self).__init__("EDU", children)

    @property
    def text(self):
        return "".join(s.text for s in self.iterfind(filter=node_type_filter([TEXT, Connective])))


class Sentence(BaseNode):
    def __init__(self, children=None, sid=None):
        children = children or []
        super(Sentence, self).__init__("SENTENCE", children)
        self.sid = sid

    @property
    def text(self):
        return "".join(s.text for s in self.iterfind(filter=node_type_filter([TEXT, Connective])))


class Relation(BaseNode):
    NN = "NN"
    NS = "NS"
    SN = "SN"

    def __init__(self, children=None, connective_ids=None, nuclear=None, ftype=None, ctype=None):
        children = children or []
        super(Relation, self).__init__("RELATION", children)
        self.connective_ids = connective_ids
        self.nuclear = nuclear
        self.ftype = ftype
        self.ctype = ctype

    def label(self):
        driven = "-".join(self.connective_ids) if self.connective_ids else None
        return "RELATION.%s.%s.(%s).DRIVEN.%s" % (self.nuclear, self.ftype, self.ctype, driven)

    def sentential(self):
        try:
            next(self.iterfind(filter=node_type_filter(Sentence)))
            return False
        except StopIteration:
            return True


class Paragraph(BaseNode):
    def __init__(self, children=None):
        children = children or []
        super(Paragraph, self).__init__("PARAGRAPH", children)

    def root_relation(self):
        try:
            return next(self.iterfind(filter=node_type_filter(Relation), prior=True))
        except StopIteration:
            return None

    def sentences(self):
        yield from self.iterfind(filter=node_type_filter(Sentence))

    def edus(self):
        yield from self.iterfind(filter=node_type_filter(EDU))

    def connectives(self):
        yield from self.iterfind(filter=node_type_filter(Connective))


class Headline(Paragraph):
    def __init__(self, children=None):
        super(Headline, self).__init__(children)
        self.set_label("HEADLINE")


class Discourse(list):
    def __init__(self, paragraphs=None, tag=None, docid=None, docdate=None):
        paragraphs = paragraphs or []
        self.tag = tag
        self.docid = docid
        self.docdate = docdate
        super(Discourse, self).__init__(paragraphs)

    @classmethod
    def from_xml(cls, file, encoding="UTF-8"):
        with open(file, "r", encoding=encoding) as fd:
            dom_doc = ElementTree.fromstring(fd.read())  # type: ElementTree.Element
        if dom_doc.tag != "DOC":
            return None
        docid = dom_doc.attrib["ID"] if "ID" in dom_doc.attrib else None
        docdate = dom_doc.attrib["DATE"] if "DATE" in dom_doc.attrib and dom_doc.attrib["DATE"] else None
        tag = file
        paragrapgs = cls._deserialize(dom_doc)
        return cls(paragrapgs, tag, docid, docdate)

    @classmethod
    def _deserialize(cls, dom_root):
        nodes = []
        for dom_node in dom_root:
            if not isinstance(dom_node, ElementTree.Element):
                continue
            if dom_node.tag == "HEADLINE":
                node_child = cls._deserialize(dom_node)
                node = Headline(node_child)
            elif dom_node.tag == "PARAGRAPH":
                node_child = cls._deserialize(dom_node)
                node = Paragraph(node_child)
            elif dom_node.tag == "RELATION":
                node_child = cls._deserialize(dom_node)
                ftype = dom_node.attrib["TYPE"] if "TYPE" in dom_node.attrib and dom_node.attrib["TYPE"] else None
                ctype = dom_node.attrib["CTYPE"] if "CTYPE" in dom_node.attrib and dom_node.attrib["CTYPE"] else None

                if ftype and (not ctype):
                    ctype = rev_relationmap[ftype]
                if "CONNECTIVES" in dom_node.attrib and dom_node.attrib["CONNECTIVES"]:
                    connective_ids = dom_node.attrib["CONNECTIVES"].split("-")
                else:
                    connective_ids = None
                if "NUCLEAR" in dom_node.attrib and dom_node.attrib["NUCLEAR"]:
                    nuclear = dom_node.attrib["NUCLEAR"]
                    nuclear = nuclearmap[nuclear]
                else:
                    nuclear = None
                node = Relation(node_child, connective_ids, nuclear, ftype, ctype)
            elif dom_node.tag == "SENTENCE":
                node_child = cls._deserialize(dom_node)
                sid = dom_node.attrib["ID"] if "ID" in dom_node.attrib and dom_node.attrib["ID"] else None
                node = Sentence(node_child, sid)
            elif dom_node.tag == "EDU":
                node_child = cls._deserialize(dom_node)
                node = EDU(node_child)
            elif dom_node.tag == "TEXT":
                node = TEXT(dom_node.text)
            elif dom_node.tag == "CONNECTIVE":
                cid = dom_node.attrib["ID"]
                node = Connective(dom_node.text, cid)
            else:
                raise ValueError("unhandled dom node")
            nodes.append(node)
        return nodes

    def to_xml(self, file, encoding="UTF-8"):
        dom_doc = ElementTree.Element("DOC")
        dom_doc.set("ID", self.docid or "")
        dom_doc.set("DATE", self.docdate or "")
        dom_child = self._serialize(self)
        dom_doc.extend(dom_child)
        with open(file, "w", encoding=encoding) as fd:
            s = ElementTree.tostring(dom_doc).decode()
            fd.write(minidom.parseString(s).toprettyxml())

    def _serialize(self, root):
        elements = []
        for node in root:
            if isinstance(node, Headline):
                element = ElementTree.Element("HEADLINE")
                element.extend(self._serialize(node))
            elif isinstance(node, Paragraph):
                element = ElementTree.Element("PARAGRAPH")
                element.extend(self._serialize(node))
            elif isinstance(node, Relation):
                element = ElementTree.Element("RELATION")
                element.set("TYPE", node.ftype or "")
                element.set("CTYPE", node.ctype or "")
                element.set("CONNECTIVES", "-".join(node.connective_ids) if node.connective_ids else "")
                element.set("NUCLEAR", rev_nuclearmap[node.nuclear] if node.nuclear else "")
                element.extend(self._serialize(node))
            elif isinstance(node, Sentence):
                element = ElementTree.Element("SENTENCE")
                element.set("ID", node.sid or "")
                element.extend(self._serialize(node))
            elif isinstance(node, EDU):
                element = ElementTree.Element("EDU")
                element.extend(self._serialize(node))
            elif isinstance(node, TEXT):
                element = ElementTree.Element("TEXT")
                element.text = node.text
            elif isinstance(node, Connective):
                element = ElementTree.Element("CONNECTIVE")
                element.set("ID", node.cid)
                element.text = node.text
            else:
                raise ValueError("unhandled node type")
            elements.append(element)
        return elements


class DependencyNode:
    def __init__(self, head=None, arc_type=None, deps=None):
        self.head = head
        self.arc_type = arc_type
        self.deps = deps or []

    def __hash__(self):
        return id(self)


class DependencyGraph:
    ROOT_ARC_TYPE = "ROOT_ARC"

    def __init__(self, tokens):
        self.root = DependencyNode()
        self.tokens = [self.root] + tokens

    def add_arc(self, head: DependencyNode, dependent: DependencyNode, arc_type=None):
        if dependent is self.root:
            raise ValueError("root node can not be used as modifier")
        if dependent.head is not None:
            raise ValueError("modifier node already has a head node")
        if dependent in head.deps:
            raise ValueError("modifier node found already the head node's modifier")

        if head is self.root:
            arc_type = self.ROOT_ARC_TYPE
        head.deps.append(dependent)
        head.deps.sort(key=self.index)
        dependent.head = head
        dependent.arc_type = arc_type

    def index(self, node):
        if node is None:
            return -1
        else:
            return self.tokens.index(node)

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        for token in self.tokens:
            yield token

    def __getitem__(self, i):
        return self.tokens[i]


class SentenceDepWrapper(DependencyNode):
    def __init__(self, sentence, head=None, arc_type=None, deps=None, subgraph=None):
        self.sentence = sentence
        self.subgraph = subgraph
        super(SentenceDepWrapper, self).__init__(head, arc_type, deps)


class EDUDepWrapper(DependencyNode):
    def __init__(self, edu, head=None, arc_type=None, deps=None):
        self.edu = edu
        super(EDUDepWrapper, self).__init__(head, arc_type, deps)


class DependencyParagraph(DependencyGraph):
    @classmethod
    def from_tree(cls, paragraph, sentence_restrict=False, left_heavy=True):
        tree_root = paragraph.root_relation()
        if tree_root is None:
            raise ValueError("No relation found in given paragraph")

        if not sentence_restrict:
            return cls.make_dependency_edus(tree_root)
        else:
            if tree_root.sentential():
                # 如果根节点就是句内关系，找到关系节点的祖先节点中的句子节点
                parent = tree_root.parent()
                while parent and not isinstance(parent, Sentence):
                    parent = parent.parent()
                if not isinstance(parent, Sentence):
                    raise ValueError("root relation not in any sentence")
                graph = cls([SentenceDepWrapper(parent)])
                graph.add_arc(graph.root, graph.tokens[1])
                graph.tokens[1].subgraph = cls.make_dependency_edus(tree_root, left_heavy)
                return graph
            else:
                graph = cls.make_dependency_sents(tree_root, left_heavy)
                for token in graph.tokens:
                    if token is not graph.root:
                        token.subgraph = cls.make_dependency_edus(token.sentence, left_heavy)
                return graph

    @classmethod
    def make_dependency_edus(cls, tree_root, left_heavy=True):
        edus = list(tree_root.iterfind(node_type_filter(EDU)))
        tokens = [EDUDepWrapper(edu) for edu in edus]
        tree_dep_maps = {edu: token for edu, token in zip(edus, tokens)}
        graph = cls(tokens)
        tree_dep_root = cls._make_dependency_edus(graph, tree_root, tree_dep_maps, left_heavy)
        graph.add_arc(graph.root, tree_dep_maps[tree_dep_root])
        return graph

    @classmethod
    def make_dependency_sents(cls, tree_root, left_heavy=True):
        sents = list(tree_root.iterfind(node_type_filter(Sentence)))
        tokens = [SentenceDepWrapper(sent) for sent in sents]
        tree_dep_maps = {sent: token for sent, token in zip(sents, tokens)}
        graph = cls(tokens)
        tree_dep_root = cls._make_dependency_sents(graph, tree_root, tree_dep_maps, left_heavy)
        graph.add_arc(graph.root, tree_dep_maps[tree_dep_root])
        return graph

    @staticmethod
    def _make_dependency_edus(graph: DependencyGraph, root, tree_dep_maps, left_heavy=True):
        if isinstance(root, EDU):
            return root
        elif isinstance(root, Sentence):
            children = []
            for child in root:
                children.append(DependencyParagraph._make_dependency_edus(graph, child, tree_dep_maps))
            nuclear = DEFAULT_NUCLEAR
            arc_type = (DEFAULT_CTYPE, DEFAULT_FTYPE, DEFAULT_NUCLEAR)
        elif isinstance(root, Relation):
            children = []
            for child in root:
                children.append(DependencyParagraph._make_dependency_edus(graph, child, tree_dep_maps))
            nuclear = root.nuclear
            if nuclear == Relation.NN:
                arc_type = (root.ctype, root.ftype, root.nuclear)
            else:
                arc_type = (root.ctype, root.ftype)
        else:
            raise ValueError("unhandled node type")

        if len(children) == 1:
            return children[0]
        elif len(children) == 2:
            if nuclear == Relation.NS or nuclear == Relation.NN:
                head = children[0]
                modifier = children[1]
            else:
                head = children[1]
                modifier = children[0]
            graph.add_arc(tree_dep_maps[head], tree_dep_maps[modifier], arc_type)
            return head
        else:
            if not left_heavy:
                children.reverse()
            _prev = None
            for child in children:
                if _prev:
                    graph.add_arc(tree_dep_maps[_prev], tree_dep_maps[child], arc_type)
                _prev = child
            return children[0]

    @staticmethod
    def _make_dependency_sents(graph, root, tree_dep_maps, left_heavy=True):
        if isinstance(root, Sentence):
            return root
        elif isinstance(root, Relation):
            children = []
            for child in root:
                children.append(DependencyParagraph._make_dependency_sents(graph, child, tree_dep_maps))
            nuclear = root.nuclear
            if nuclear == Relation.NN:
                arc_type = (root.ctype, root.ftype, root.nuclear)
            else:
                arc_type = (root.ctype, root.ftype)
        else:
            raise ValueError("unhandled node type")

        if len(children) == 1:
            return children[0]
        elif len(children) == 2:
            if nuclear == Relation.NS or nuclear == Relation.NN:
                head = children[0]
                modifier = children[1]
            else:
                head = children[1]
                modifier = children[0]
            graph.add_arc(tree_dep_maps[head], tree_dep_maps[modifier], arc_type)
            return head
        else:
            if not left_heavy:
                children.reverse()
            _prev = None
            for child in children:
                if _prev:
                    graph.add_arc(tree_dep_maps[_prev], tree_dep_maps[child], arc_type)
                _prev = child
            return children[0]

    def to_tree(self):
        # TODO
        raise NotImplementedError()

    def __str__(self):
        token_reprs = []
        for token in self.tokens:
            if token is self.root:
                token_repr = "ROOT"
            elif isinstance(token, EDUDepWrapper):
                token_repr = "".join(token.edu.text)
            elif isinstance(token, SentenceDepWrapper):
                token_repr = "".join(token.sentence.text)
            else:
                token_repr = str(token).replace("\n", "")
            if len(token_repr) > 10:
                token_repr = token_repr[:10] + "..."
            token_reprs.append(token_repr)
        ret_repr = ""
        for token, repr_str in zip(self.tokens, token_reprs):
            ret_repr += "%-3d %-30s %-3d %s" % \
                        (self.index(token), repr_str, self.index(token.head), str(token.arc_type))\
                        + "\n"
        return ret_repr


def node_type_filter(types):
    if not isinstance(types, list):
        types = [types]

    def filter(node):
        return any([isinstance(node, t) for t in types])
    return filter


nuclearmap = {"LEFT": Relation.NS, "RIGHT": Relation.SN, "ALL": Relation.NN}
rev_nuclearmap = {v: k for k, v in nuclearmap.items()}
relationmap = {'因果类': ['因果关系', '推断关系', '假设关系', '目的关系', '条件关系', '背景关系'],
               '并列类': ['并列关系', '顺承关系', '递进关系', '选择关系', '对比关系'],
               '转折类': ['转折关系', '让步关系'],
               '解说类': ['解说关系', '总分关系', '例证关系', '评价关系']}
rev_relationmap = {}
for coarse_class, fine_classes in relationmap.items():
    rev_relationmap.update((sub_class, coarse_class) for sub_class in fine_classes)

DEFAULT_NUCLEAR = Relation.NN
DEFAULT_CTYPE = "并列类"
DEFAULT_FTYPE = "并列关系"
