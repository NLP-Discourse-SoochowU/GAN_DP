# coding: UTF-8
import subprocess
import thulac
import threading
import os
from nltk.tree import Tree


BERKELEY_JAR = "berkeleyparser/BerkeleyParser-1.7.jar"
BERKELEY_GRAMMAR = "berkeleyparser/chn_sm5.gr"


class BerkeleyParser(object):
    def __init__(self):
        self.tokenizer = thulac.thulac()
        self.cmd = ['java', '-Xmx1024m', '-jar', BERKELEY_JAR, '-gr', BERKELEY_GRAMMAR]
        self.process = self.start()

    def start(self):
        return subprocess.Popen(self.cmd, env=dict(os.environ), universal_newlines=True, shell=False, bufsize=0,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, errors='ignore')

    def stop(self):
        if self.process:
            self.process.terminate()

    def restart(self):
        self.stop()
        self.process = self.start()

    def parse_thread(self, text, results):
        text = text.replace("(", '-LRB-')
        text = text.replace(")", '-RRB-')
        self.process.stdin.write(text + '\n')
        self.process.stdin.flush()
        ret = self.process.stdout.readline().strip()
        results.append(ret)

    def parse(self, words, timeout=20000):
        # words, _ = list(zip(*self.tokenizer.cut(text)))
        results = []
        t = threading.Thread(target=self.parse_thread, kwargs={'text': " ".join(words), 'results': results})
        t.setDaemon(True)
        t.start()
        t.join(timeout)

        if not results:
            self.restart()
            raise TimeoutError()
        else:
            return Tree.fromstring(results[0])
