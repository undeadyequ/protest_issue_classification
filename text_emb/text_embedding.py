import torch
from abc import ABC
from abc import abstractmethod
from typing import Iterable, Union
#from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from torchnlp.word_to_vector import FastText
import pandas as pd
import numpy as np


class TextEmbedding(ABC):
    @abstractmethod
    def embedtext(self, text: Iterable) -> Union[torch.Tensor, np.array]:
        raise NotImplementedError


class Fasttext(TextEmbedding):
    pretrained_fasttext = "/home/Data/pretrained_model"
    def __init__(self):
        self.fasttext_emb = FastText(cache=self.pretrained_fasttext)
    def embedtext(self, text: Iterable) -> Union[torch.Tensor, np.array]:
        if isinstance(text, np.ndarray):
            text = text.tolist()
        return self.fasttext_emb[text]


class Tfidf(TextEmbedding):
    def __init__(self,
                 id_trans_f,
                 sublinear_tf=True,
                 min_df=2,
                 max_df=30,
                 norm='l2',
                 encoding='utf-8',
                 ngram_range=(1, 2),
                 stop_words='english'):
        self.tfidf_emb = TfidfVectorizer(sublinear_tf=sublinear_tf,
                                     min_df=min_df,
                                     max_df=max_df,
                                     norm=norm,
                                     encoding=encoding,
                                     ngram_range=ngram_range,
                                     stop_words=stop_words)

        self.id_lab_trans = pd.read_csv(id_trans_f, delimiter=",")
        trans = self.id_lab_trans["trans"]
        corp_fts = self.tfidf_emb.fit_transform(trans.values.astype('U')).toarray()
    def embedtext(self, text: Iterable) -> Union[torch.Tensor, np.array]:
        return self.tfidf_emb.transform(text)


class BOW(TextEmbedding):
    def __init__(self,
                 id_trans_f,
                 stop_words="english"):
        self.id_lab_trans = pd.read_csv(id_trans_f, delimiter=",")
        trans = self.id_lab_trans["trans"]

        self.bow_emb = CountVectorizer(stop_words=stop_words)
        self.bow_emb.fit_transform(trans.values.astype('U')).toarray()
    def embedtext(self, text: Iterable) -> Union[torch.Tensor, np.array]:
        return self.bow_emb.transform(text)


#### Sentence Embedding
