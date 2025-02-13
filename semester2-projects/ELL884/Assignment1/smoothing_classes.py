from ngram import NGramBase
from config import *
import numpy as np
import pandas as pd

class NoSmoothing(NGramBase):

    def __init__(self):

        super(NoSmoothing, self).__init__()
        self.update_config(no_smoothing)

class AddK(NGramBase):

    def __init__(self):
        self.update_config(add_k)

class StupidBackoff(NGramBase):

    def __init__(self):
        self.update_config(stupid_backoff)

class GoodTuring(NGramBase):

    def __init__(self):
        self.update_config(good_turing)

class Interpolation(NGramBase):

    def __init__(self):
        self.update_config(interpolation)

class KneserNey(NGramBase):

    def __init__(self):
        self.update_config(kneser_ney)


if __name__=="__main__":
    ns = NoSmoothing()
    ns.method_name()
